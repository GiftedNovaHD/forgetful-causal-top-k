from torch.nn.attention.flex_attention import or_masks, and_masks, _mask_mod_signature

"""
  Notes for coding masks with flex attention:
    - The 2 tokens will be allowed to attention to eachother if the mask returns True
    - Logical operations need to use bitwise operators
"""

def inverse_mask(mask: _mask_mod_signature) -> _mask_mod_signature:
  """
    Return a mask_mod that's the inverse of the provided mask
  """
  if not callable(mask):
    raise RuntimeError(f"{mask} is not a callable mask mod")
  
  def inverse_mask(b, h, q_idx, kv_idx):
    result =  ~ mask(b, h, q_idx, kv_idx)

    return result
  
  return inverse_mask

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

def persistent_mask_constructor(num_persistent_tokens):
  """
    Generate a mask for persistent tokens.
  
    Args:
      num_persistent_tokens: int, the number of persistent tokens
  """
  def persistent(b, h, q_idx, kv_idx):
    return kv_idx < num_persistent_tokens
  
  return persistent

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
  
  swap_mask = swa_persistent_mask_constructor(window_size, num_persistent_tokens)

  inverse_swap_mask = inverse_mask(swap_mask)

  top_k_mask = and_masks(inverse_swap_mask, causal_mask)

  return top_k_mask