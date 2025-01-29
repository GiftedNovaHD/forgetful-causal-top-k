import torch.nn as nn
import lorann
import lorann_gpu
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import _score_mod_signature

    
class ForgetfulCausalTopKAttention(nn.Module): 
  def __init__(self, 
               embed_dim, 
               latent_dim, 
               num_heads, 
               k,
               forget_prob=0.1, 
               window_size=None, 
               persistent_tokens=4): 
    """ 
    By default, we use 8-bit integer quantization for lower memory consumption and for LoRANN compatability
    """
    super().__init__()
    self.embed_dim = embed_dim
    self.latent_dim = latent_dim
    self.num_heads = num_heads
    self.k = k
    self.forget_prob = forget_prob
    self.window_size = window_size
    self.persistent_tokens = persistent_tokens

    assert self.head_dim * num_heads == latent_dim, "latent_dim must be divisible by num_heads"

    # Projection layers
    self.q_proj = nn.Linear(embed_dim, latent_dim)
    self.k_proj = nn.Linear(embed_dim, latent_dim)
    self.v_proj = nn.Linear(embed_dim, latent_dim)
    self.output_proj = nn.Linear(latent_dim, embed_dim)

    # Learnable persistent tokens (your attention sinks) 
    if persistent_tokens > 0: 
      self.persistent_k = nn.Parameter(torch.randn(1, persistent_tokens, latent_dim))
      self.persistent_v = nn.Parameter(torch.randn(1, persistent_tokens, latent_dim))

  def forward(self, x, mask=None): 
    batch_size, seq_len, _ = x.shape
    original_seq_len = seq_len

    # Project queries, key, values, like in Deepseek V2 and V3 et. al. 
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # Add persistent tokens 
    if self.persistent_tokens > 0: 
      persistent_k = self.persistent_k.expand(batch_size, -1, -1)
      persistent_v = self.persistent_v.expand(batch_size, -1, -1)
      k = torch.cat([k, persistent_k], dim=1)
      v = torch.cat([v, persistent_v], dim=1)
      seq_len_k = seq_len + self.persistent_tokens

    else: 
      seq_len_k = seq_len

    # Reshape for multi-head latent attention
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

    # Compute attention scores 
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5) 

    # Apply the combined mask 
    causal_mask = self._create_causal_mask(original_seq_len, seq_len_k, x.device)
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Apply sliding window mask 
    if self.window_size is not None: 
      window_mask = self._create_window_mask(original_seq_len, seq_len_k, x.device)
      attn_scores = attn_scores.masked_fill(~window_mask, float('-inf'))
    
    # Top-k selection and forgetful causal masking
    attn_weights = self._apply_topk_masking(attn_scores)

    # Attention output
    output = torch.matmul(attn_weights, v)
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.latent_dim)
    output = self.output_proj(output)

  def _create_causal_mask(self, seq_len, seq_len_k, device): 
    mask = torch.ones((seq_len, seq_len_k), dtype=torch.bool, device=device)
    if self.persistent_tokens > 0: 
      # Allow full attention to persistent tokens
      mask[:, :self.persistent_tokens] = False
      # Causal masking for original sequence positions
      causal_part = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1).bool()
      mask[:, self.persistent_tokens:] = causal_part
    else: 
      mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]

  def _create_window_mask(self, seq_len, seq_len_k, device): 
    mask = torch.zeros((seq_len, seq_len_k), dtype=torch.bool, device=device)
    for i in range(seq_len): 
      # Always attend to persistent tokens 
      mask[i, :self.persistent_tokens] = True
      # Sliding window for sequence tokens
      start = max(self.persistent_tokens, 
                  self.persistent_tokens + i - self.window_size + 1)
      end = self.persistent_tokens + i + 1 
      mask[i, start:end] = True
    return mask[None, None, :, :] 
  
  def _apply_topk_masking(self, scores): 
    # Select top-k values 
    topk_values, topk_indices = torch.topk(scores, k=self.k, dim=-1)

    # Create mask and apply forgetful masking 
    topk_mask = torch.zeros_like(scores, dtype=torch.bool)
    topk_mask.scatter_(dim=-1, index=topk_indices)

    if self.training and self.forget_prob > 0: 
      forget_mask = torch.rand_like(topk_values) < self.forget_prob
      topk_values = topk_values.masked_fill(forget_mask, float('-inf'))

    # Combine masks 
    masked_scores = torch.full_like(scores, float('-inf'))
    masked_scores.masked_scatter_(topk_mask, topk_values)

    return F.softmax(masked_scores, dim=-1)
