import torch.nn as nn
from torch.nn.attention.flex_attention import _score_mod_signature, or_masks, and_masks

"""
  Notes for coding masks with flex attention:
    - The 2 tokens will be allowed to attention to eachother if the mask returns True
"""

def sliding_window_mask(window_size):
  """
    Generate a mask for sliding window attention given a window size.

    Args:
      window_size: int, the size of the sliding window
  """

  def sliding_window(b, h, q_idx, kv_idx):
    return q_idx - kv_idx <= window_size
  
  return sliding_window
  

def causal_mask(b, h, q_idx, kv_idx):
  return q_idx >= kv_idx

def persistent_mask(num_persistent_tokens):
  """
    Generate a mask for persistent tokens.
  
    Args:
      num_persistent_tokens: int, the number of persistent tokens
  """
  def persistent(b, h, q_idx, kv_idx):
    return kv_idx <= num_persistent_tokens
  
  return persistent

def swa_persistent_mask(window_size, num_persistent_tokens):
  """
    Generate a mask for sliding window attention with persistent tokens.
  Args:
      window_size: int, the size of the sliding window
      num_persistent_tokens: int, the number of persistent tokens
  """
  swa_mask = sliding_window_mask(window_size)
  persistent_mask = persistent_mask(num_persistent_tokens)

  swa_persistent = or_masks(swa_mask, persistent_mask)

  return swa_persistent