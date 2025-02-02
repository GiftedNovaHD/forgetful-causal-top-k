import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl

#### MLA Kernel ####
# We assume that the keys and values are already projected into a latent space with a fixed number of latent tokens per batch and that our queries
# are in full space. 
# This is just a trial implementation
# TODO: Optimize inner loops and do 
# TODO: More aggressive operator fusion 
# TODO: Add support for topk causal mask 

@triton.jit 
def mla_kernel_fwd(
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


@triton.jit
def mla_kernel_bwd(
  dO_ptr,       # Pointer to gradient of output
  Q_ptr, K_ptr, V_ptr, Wq_ptr, # Pointers to original input tensors Q (B, L, Dq), K (B, L, Dk),  V (B, L, Dv), and projection matrix for Q (Dq, Dk)
  dQ_ptr, dK_ptr, dV_ptr, # Pointers to gradient output tensors dQ (B, L, Dq), dK (B, L, Dk), dV (B, L, Dv)
  dWq_ptr, # Pointer to gradient output for Wq (Dq, Dk)
  B: tl.constexpr, # Batch size
  L: tl.constexpr, # Sequence length
  Dq: tl.constexpr, # Query dimension (increased)
  Dk: tl.constexpr, # Dimension of K and projected Q
  Dv: tl.constexpr, # Dimension of V (compressed)
  BLOCK_M: tl.constexpr, # Block size over queries
  BLOCK_N: tl.constexpr, # Block size over keys
  ): 
  """
  Backward kernel for Multi-Head Latent Attention with DiffQKV (differentiated rescaling of query, key, and value)

  1. recompute Q_proj = Q * Wq for the current block 
  2. Compute attention scores S = Q_proj @ K^{\intercal} and the softmax weights A 
    - TODO: Load saved intermediaries from the forward pass if available, i.e. in the spirit of FA-2
    - NOTE: For now we recomputed them below
  3. Compute gradients with respect to softmax input S. 
    - Out = A @ V
    - dO is the gradient of Out. 
    - dS = A ⊙ ( (dO * V^T) - sum( (dO * V^T) ⊙ A, axis=key_dim) )
    - Note that the gradient computation for softmax is nontrivial because the Jacobian of SoftMax is dA/dS = diag(A) - A A^{\intercal}
    - We use this to compute dS from dO and V. 
  4. Backprop through the dot-product S = Q_proj @ K^{\intercal}
    - Compute gradients for Q_proj and K 
  5. Backprop through the projection matrix Wq, 
    - Q_proj = Q @ Wq
    - dQ = dQ_proj @ Wq^{\intercal}
    - dWq = Q^{\intercal} @ dQ_proj
  6. Backprop through the weighted sum to compute dV
  """
  b = tl.program_id(0) 
  m_start = tl.program_id(1) * BLOCK_M

  q_proj = tl.zeros((BLOCK_M, Dk), dtype=tl.float32)
  
  # Step 1 
  for i in range(BLOCK_M): 
    m = m_start + i 
    if m < L:
      # Load Q vector for query m (length Dq) 
      q_vec = tl.load(Q_ptr + b * L * Dq + m * Dq, mask=tl.arange(0, Dq) < Dq, other=0.0)
      # Compute Q_proj = Q @ Wq
      for j in range(Dk): 
        acc = 0.0
        for d in range(Dq): 
          # Wq is stored row-major 
          acc += q_vec[d] * tl.load(Wq_ptr + d * Dk + j)
        q_proj[i, j] = acc
    else: 
      for j in range(Dk): 
        q_proj[i, j] = 0.0

  # Step 2 Compute attention scores S = Q_proj @ K^{\intercal}
  max_logits = tl.zeros((BLOCK_M,), dtype=tl.float32)

  for n in range(0, L, BLOCK_N): 
    for j in range(BLOCK_N): 
      key_index = n + j 
      for i in range(BLOCK_M): 
        if key_index < L: 
          dot = 0.0
          for d in range(Dk): 
            q_val = q_proj[i, d] 
            k_val = tl.load(K_ptr + b * L * Dk + key_index * Dk + d) 
            dot += q_val * k_val
          max_logits[i] = tl.maximum(max_logits[i], dot)
  
  # TODO: Use saved intermediaries from the forward pass if available
  sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
  # Allocate storage for the softmax weights for the block
  A_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

  # Store dot product values for backprop
  S_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

  for n in range(0, L, BLOCK_N): 
    for j in range(BLOCK_N): 
      key_index = n + j 
      for i in range(BLOCK_M): 
        if key_index < L: 
          dot = 0.0
          for d in range(Dk):
            dot += q_proj[i, d] * tl.load(K_ptr + b * L * Dk + key_index * Dk + d)
          S_block[i, j] = dot 
          exp_val = tl.exp(dot - max_logits[i])
          A_block[i, j] = exp_val
          sum_exp[i] += exp_val
        else: 
          A_block[i, j] = 0.0
          S_block[i, j] = 0.0

  # Normalize to get softmax weights A 
  for i in range(BLOCK_M): 
    for j in range(BLOCK_N): 
      A_block[i, j] /= (sum_exp[i] + 1e-6)
  
  # Step 3
  dS_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  X = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for n in range(0, L, BLOCK_N):
    for j in range(BLOCK_N): 
      key_index = n + j 
      for i in range(BLOCK_M):
        if key_index < L:
          # Compute dot product between dO and V for this key 
          dot_dO_V = 0.0 
          for d in range(Dv): 
            dO_val = tl.load(dO_ptr + b * L * Dv + (m_start + i) * Dv + d)
            v_val = tl.load(V_ptr + b * L * Dv + key_index * Dv + d) 
            dot_dO_V += dO_val * v_val
          X[i, j] = dot_dO_V
        else: 
          X[i, j] = 0.0

  # Compute weighted sum of X over keys per query for the softmax gradient 
  sum_AX = tl.zeros((BLOCK_M,), dtype=tl.float32)
  for i in range(BLOCK_M): 
    for j in range(BLOCK_N): 
      sum_AX[i] += A_block[i, j] * X[i, j]

  for i in range(BLOCK_M): 
    for j in range(BLOCK_N): 
      dS_block[i, j] = A_block[i, j] * (X[i, j] - sum_AX[i])

  # Step 4 
  # Create a buffer to store gradients of Q_proj
  dQ_proj = tl.zeros((BLOCK_M, Dk), dtype=tl.float32)
  # Accumulate gradients for K into a temporary buffer
  dK_temp = tl.zeros((BLOCK_N, Dk), dtype=tl.float32)

  for n in range(0, L, BLOCK_M): 
    for j in range(BLOCK_N): 
      key_index = n + j 
      for i in range(BLOCK_M): 
        if key_index < L: 
          for d in range(Dk): 
            # Gradient contribution for key_index from query i 
            grad_contrib = dS_block[i, j] * tl.load(K_ptr + b * L * Dk + key_index * Dk + d)
            # Accumulate gradient for Q_proj
            dQ_proj[i, d] += grad_contrib 
            # Accumulate gradient for K 
            dK_temp[j, d] += dS_block[i, j] * q_proj[i, d]
  
  # Write out gradients for K
  for n in range(0, L, BLOCK_M): 
    for j in range(BLOCK_N): 
      key_index = n + j 
      if key_index < L: 
        for d in range(Dk): 
          # NOTE: To implement a reduction across query blocks 
          tl.store(dK_ptr + b * L * Dk + key_index * Dk + d, dK_temp[i, j])

  # Step 5
  for i in range(BLOCK_M): 
    m = m_start + i 
    if m < L: 
      # Compute gradients for Q[m, :] 
      for d in range(Dq): 
        acc = 0.0
        for j in range(Dk): 
          # Load corresponding Wq value 
          w_val = tl.load(Wq_ptr + d * Dk + j) 
          acc += dQ_proj[i, j] * w_val 
        tl.store(dQ_ptr + b * L * Dq + m * Dq + d, acc)

  
  # Accumulate gradient for Wq
  # For each element in Wq (d, j), accumulate over queries
  for i in range(BLOCK_M): 
    m = m_start + i 
    if m < L: 
      q_vec = tl.load(Q_ptr + b * L * Dq + m * Dq, mask=tl.arange(0, Dq) < Dq, other=0.0)
      for j in range(Dk): 
        for d in range(Dq): 
          # dWq[d, j] += Q[m, d] * dQ_proj[i, j]
          # NOTE: To reduce across query blocks later 
          cur = tl.load(dWq_ptr + d * Dk + j)
          tl.store(dWq_ptr + d * Dk + j, cur + q_vec[d] * dQ_proj[i, j])
  
  # Step 6 
  # Given Out = A * V, we compute dV from dO and the softmax weights 
  # For each key index, we compute 
  # dV[b, key_index, :] = sum_over_queries( A[q, key_index] * dO[q, :])
  for n in range(0, L, BLOCK_N): 
    for j in range(BLOCK_N): 
      key_index = n + j 
      for i in range(BLOCK_M): 
        if key_index < L: 
          for d in range(Dk): 
            # Load dO for each query m and V for key_index
            dO_val = tl.load(dO_ptr + b * L * Dv + (m_start + i) * Dv + d) 
            # Accumulate gradients for V 
            cur = tl.load(dV_ptr + b * L * Dv + key_index * Dv + d) 
            tl.store(dV_ptr + b * L * Dv + key_index * Dv + d, cur + A_block[i, j] * dO_val)