"""
  Modified from: https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/mods/softcapping.py

"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature, or_masks, and_masks

# Some internal torch.compile details
from torch._inductor.virtualized import ops
from torch._inductor.lowering import make_pointwise, register_lowering
from functools import partial


@torch.library.custom_op("approx::tanh", mutates_args=())
def _tanh_approx(inp: Tensor) -> Tensor:
    return torch.tanh(inp)


@_tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


def _tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)


register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)


class _TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)

    @staticmethod
    def vmap(info, in_dims, x):
        return torch.tanh(x), 0


_tanh_approx = _TanhApprox.apply


def tanh_softcap_constructor(soft_cap: int, approx: bool = False) -> _score_mod_signature:
    """Returns an tanh bias score_mod given the number of heads H

    Args:
        soft_cap: The soft cap value to use for normalizing logits
        approx: Whether to use the `tanh.approx.` ptx instruction

    Returns:
        tanh_softcap: score_mod
    """
    tanh = _tanh_approx if approx else torch.tanh

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return soft_cap * tanh(score / soft_cap)

    prefix = "tanh_softcap_approx" if approx else "tanh_softcap"
    tanh_softcap.__name__ = f"{prefix}_{soft_cap}"

    return tanh_softcap

def dynamic_topk_softmax_constructor(threshold: float):
  """
    Returns a softmax approximation generated through dynamic top-k such that
    when the score of the lowest token is below the threshold, no additional tokens are looked at

    Args:
      threshold: The threshold value to use for the dynamic top-k softmax
  """

  def dynamic_topk_softmax(score, b, h, q_idx, kv_idx):
      raise NotImplementedError