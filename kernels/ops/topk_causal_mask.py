import torch
from torch import autograd
import torch.nn.functional as F 

import triton
import triton.language as tl

@triton.jit 
def topk_kernel(
  scores_ptr, 
  topk_indices_ptr, 
  B: tl.constexpr, 
  L_q: tl.constexpr,
  L_k: tl.constexpr,
  k: tl.constexpr,
  ): 
  b = tl.program_id(0)
  q_id = tl.program_id(1) # Each program handles one query row-wise

  # Load scores for given query row 
  scores_offset = b * L_q * L_k + q_id * L_k 
  scores = tl.load(scores_ptr + scores_offset, mask=tl.arange(0, L_k) < L_k, other=-1e9)

  # Allocate space for top-k values and indices in registers / SMEM 
  topk_values = tl.full((k,), -1e9, dtype=scores.dtype)
  topk_indices = tl.zeros((k,), dtype=tl.int32)

  # Loop over the keys to find top-k values and indices 
  # Future: Optimize using heap / bitonic sort 
  for k_idx in range(L_k): 
    score = tl.load(scores_ptr + scores_offset + k_idx)
    # Compare with smallest value in topk_vals 
    min_val = tl.min(topk_values)
    min_idx = tl.argmin(topk_values)

    if score > min_val: 
      topk_values[min_idx] = score
      topk_indices[min_idx] = k_idx 

  # Write the top-k values and indices to global memory
  output_offset = b * L_q * k + q_id * k
  tl.store(topk_indices_ptr + output_offset, topk_indices)

@triton.jit
def causal_mask_kernel(
  scores_ptr, 
  B: tl.constexpr, 
  L_q: tl.constexpr,
  L_k: tl.constexpr,
  ):
  b = tl.program_id(0)
  q = tl.program_id(1) 
  key_idx = tl.arange(0, L_k)

  # For each query index, compute the mask 
  mask = key_idx <= q

  offset = b * L_q * L_k + q * L_k 
  scores = tl.load(scores_ptr + offset, mask=tl.arange(0, L_k) < L_k, other=-1e9)

  masked_scores = tl.where(mask, scores, -1e9)
  tl.store(scores_ptr + offset, masked_scores)