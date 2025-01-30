import torch.nn as nn
from torch.nn.attention.flex_attention import or_masks, and_masks

"""
  Notes for coding masks with flex attention:
    - The 2 tokens will be allowed to attention to eachother if the mask returns True
"""

def causal_mask(b, h, q_idx, kv_idx):
  return q_idx >= kv_idx

def sliding_window_mask_constructor(window_size):
  """
    Generate a mask for sliding window attention given a window size.

    Args:
      window_size: int, the size of the sliding window
  """

  def sliding_window(b, h, q_idx, kv_idx):
    return q_idx - kv_idx < window_size
  
  sliding_window_mask = and_masks(sliding_window,causal_mask)
  
  return sliding_window_mask

def inverse_sliding_window_mask_constructor(window_size):
  """
    Generate an inverse mask for sliding window attention given a window size.
  """

  def inverse_sliding_window(b, h, q_idx, kv_idx):
    return q_idx - kv_idx >= window_size
  
  inverse_sliding_window_mask = and_masks(inverse_sliding_window,causal_mask)

  return inverse_sliding_window_mask

def persistent_mask_constructor(num_persistent_tokens):
  """
    Generate a mask for persistent tokens.
  
    Args:
      num_persistent_tokens: int, the number of persistent tokens
  """
  def persistent(b, h, q_idx, kv_idx):
    return kv_idx < num_persistent_tokens
  
  return persistent


def inverse_persistent_mask_constructor(num_persistent_tokens):
  """
    Generate an inverse mask for persistent tokens.

    Args:
      num_persistent_tokens: int, the number of persistent tokens
  """
  def inverse_persistent(b, h, q_idx, kv_idx):
    return kv_idx >= num_persistent_tokens

  return inverse_persistent

def swa_persistent_mask_constructor(window_size, num_persistent_tokens):
  """
    Generate a mask for sliding window attention with persistent tokens.
  Args:
      window_size: int, the size of the sliding window
      num_persistent_tokens: int, the number of persistent tokens
  """
  swa_mask = sliding_window_mask_constructor(window_size)
  persistent_mask = persistent_mask_constructor(num_persistent_tokens)

  swa_persistent = or_masks(swa_mask, persistent_mask)

  return swa_persistent

def top_k_mask_constructor(window_size, num_persistent_tokens, ):
  """
    Generate a mask for tokens which top_k attention will be applied to
  Args:
      window_size: int, the size of the sliding window
      num_persistent_tokens: int, the number of persistent tokens
  """
  
  inverse_persistent_mask = inverse_persistent_mask_constructor(num_persistent_tokens)
  inverse_sliding_window_mask = inverse_sliding_window_mask_constructor(window_size)

  top_k_mask = and_masks(inverse_persistent_mask, inverse_sliding_window_mask)

  return top_k_mask
