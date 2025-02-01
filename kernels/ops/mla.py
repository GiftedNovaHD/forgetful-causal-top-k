import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl

#### MLA Kernel ####
# We assume that the keys and values are already projected into a latent space with a fixed number of latent tokens per batch and that our queries
# are in full space. 
# This is just a trial implementation
# TODO: Optimize inner loops 
# TODO: More aggressive operator fusion 
# TODO: Add support for topk causal mask 

@triton.jit 
def mla_kernel(
  Q_ptr,                  # pointer to Q (batch_size, L, Dq)
  K_ptr,                  # pointer to K (batch_size, L, Dk)
  V_ptr,                  # pointer to V (batch_size, L, Dv)
  Wq_ptr,                 # pointer to Wq (Dq, Dk); the projection matrix for Q
  Output_ptr,             # pointer to output (batch_size, L, Dv)
  B: tl.constexpr,        # batch size
  L: tl.constexpr,        # sequence length
  Dq: tl.constexpr,       # query dimension (with increased head dimension)
  Dk: tl.constexpr,       # key dimension (and also projected Q)
  Dv: tl.constexpr,       # value dimension (compressed as per DiffQKV Implementation)
  BLOCK_M: tl.constexpr,  # block size for query dimension
  BLOCK_N: tl.constexpr,  # block size for key/value dimension
  ): 
  b = tl.program_id(0) # which batch to process
  m_start = tl.program_id(1) # starting query index within this batch

  # Allocate space for projected queries (BLOCK_M x Dk)
  q_proj = tl.zeros((BLOCK_M, Dq), dtype=tl.float32)

  # For each query in the block, we load Q into SMEM and apply Q_proj = Q @ Wq
  for i in range(BLOCK_M): 
    m = m_start + i
    if m < L: 
      # Pointer to start of Q for this query row (each row has Dq elements)
      q_offset = b * L * Dq + m * Dq
      # NOTE: to vectorize this loading in the future because we are currently doing it sequentially :clown: 
      q_vec = tl.load(Q_ptr + q_offset, mask=tl.arange(0, Dq) < Dq, other=0.0)
      # Compute vector projection: q_proj = q_vec (1, Dq) dot Wq (Dq, Dk) = (1, Dk) 
      # NOTE: this looping such a bad way to do it omg I need to unroll this in the future 
      for j in range(Dk): 
        acc = 0.0 
        for d in range(Dq):
          # Wq is stored in row-major order: Wq[d, j]
          w_val = tl.load(Wq_ptr + d * Dk + j)
          acc += q_vec[d] * w_val
        # Store the projected value 
        q_proj[i, j] = acc
      else: 
        # If we are out of bounds, we can just zero out the rest of the projected query
        for j in range(Dk): 
          q_proj[i, j] = 0.0
  
  # Now we have the projected queries in q_proj, we can compute the attention scores
  # For each query row in the block, we compute the attention scores over keys in blocks of size BLOCK_N
  # We will accumulate the softmax numerator sums for later weighted sum over V 
  
  out = tl.zeros((BLOCK_M, Dv), dtype=tl.float32) # Allocate register to store final output for each query in this block

  # Perform a two pass SoftMax for numerical stability 
  # First pass: compute the max value for each query row
  # Second pass: compute exponentials and weighted sums 

  max_logits = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)

  # Pass 1
  for n in range(0, L, BLOCK_N):
    # For each block of keys, load K into SMEM (size: BLOCK_N x Dk) 
    for j in range(BLOCK_N): 
      key_index = n + j 
      for i in range(BLOCK_M): 
        if key_index < L: 
          # Compute dot product between q_proj[i, :] and K[b, key_index, :] 
          dot = 0.0 
          for d in range(Dk): 
            q_val = q_proj[i, d]
            k_val = tl.load(K_ptr + b * L * Dk + key_index * Dk + d)
            dot += q_val * k
          # Update max logits per query for stability
          max_logits[i] = tl.maximum(max_logits[i], dot)
        else: 
          # Skip out-of-bound keys 
          pass 
  
  # Pass 2
  sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
  # Temporarily store the softmax weights for the BLOCK_N keys  
  softmax_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

  # Loop again over key blocks 
  for n in range(0, L, BLOCK_N): 
    for j in range(BLOCK_N): 
      key_index = n + j 
      for i in range(BLOCK_M): 
        if key_index < L: 
          # Compute dot product between q_proj[i, :] and K[b, key_index, :] 
          # Need to cache this later 
          dot = 0.0 
          for d in range(Dk):
            q_val = q_proj[i, d]
            k_val = tl.load(K_ptr + b * L * Dk + key_index * Dk + d)
            dot += q_val * k_val
          # Subtract max logit for numerical stability. Then exponentiate and accumulate
          exp_val = tl.exp(dot - max_logits[i])
          softmax_block[i, j] = exp_val
          sum_exp[i] += exp_val
        else: 
          softmax_block[i, j] = 0.0 

  # Use softmax weights to update the output
  for j in range(BLOCK_N): # Loop over BLOCK_N keys to perform weighted sum over V
    key_index = n + j
    for i in range(BLOCK_M): 
      if key_index < L: 
        # Normalize softmax weights for this key 
        weight = softmax_block[i, j] / (sum_exp[i] + 1e-6)
        # Load the compressed V vector 
        for d in range(Dv): 
          v_val = tl.load(V_ptr + (b * L * Dv + key_index * Dv + d))
          out[i, d] += weight * v_val 
      else: 
        # Skip out-of-bound keys
        pass 
  
  # Write computed output for each query in this block 
  for i in range(BLOCK_M): 
    m = m_start + i 
    if m < L:
      for d in range(Dv):
        tl.store(Output_ptr + (b * L * Dv + m * Dv + d), out[i, d])