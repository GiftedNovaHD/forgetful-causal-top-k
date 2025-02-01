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