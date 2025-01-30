import torch.nn as nn
import lorann
import lorann_gpu
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature

    
class ForgetfulCausalTopKAttention(nn.Module): 
  def __init__(self, 
               hidden_dim, 
               kv_latent_dim, 
               num_heads,
               forget_prob=0.1, 
               window_size=128, 
               persistent_tokens=16): 
    """ 
    By default, we use 8-bit integer quantization for lower memory consumption and for LoRANN compatability


    Args:
      hidden_dim: int, the dimension of the input and output
      kv_latent_dim: int, the dimension of the key-value latent
      num_heads: int, the number of attention heads
      forget_prob: float, the probability of forgetting a token in the top-k
      window_size: int, the size of the sliding window to consider for attention
      persistent_tokens: int, the number of tokens to function as attention sinks

    """

    super().__init__()

    self.hidden_dim = hidden_dim
    self.kv_latent_dim = kv_latent_dim
    self.num_heads = num_heads
    self.forget_prob = forget_prob
    self.window_size = window_size
    self.persistent_tokens = persistent_tokens

    self.head_dim = hidden_dim//num_heads
    
    self.scale = self.head_dim**-0.5

    self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    self.kv_latent_proj = nn.Linear(hidden_dim, kv_latent_dim, bias=False)
    self.k_proj = nn.Linear(kv_latent_dim, hidden_dim, bias=False)
    self.v_proj = nn.Linear(kv_latent_dim, hidden_dim, bias=False)

    self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

  def forward(self, x, kv_cache: Tensor = None):
    """
    Usage notes:
      - If kv_cache is None, we assume that x is of shape (batch_size, seq_len, hidden_dim)
      - If kv_cache is not None, we assume that x is of shape (batch_size, 1, hidden_dim)

    Args:
      x: Tensor, the input tensor of shape (batch_size, seq_len/1, hidden_dim)
      kv_cache: Tensor, the key-value cache of shape (batch_size, seq_len, hidden_dim)
    """