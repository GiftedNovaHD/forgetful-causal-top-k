import torch
from torch.nn.attention.flex_attention import _score_mod_signature, or_masks, and_masks

from mask import (sliding_window_mask_constructor,
                  persistent_mask_constructor,
                  swa_persistent_mask_constructor,
                  top_k_mask_constructor,
                  inverse_persistent_mask_constructor,
                  inverse_sliding_window_mask_constructor
                 )

from score_mod import (tanh_softcap_constructor)

def main(device: str = "cpu"):
    """Visualize the attention scores of sliding window mask mod.

    Args:
        device (str): Device to use for computation. Defaults
    """
    from attn_gym_utils import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 24, 8

    window_size = 4
    num_persistent_tokens = 2
    tanh_softcap = 30

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN + num_persistent_tokens, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    swa_mask = sliding_window_mask_constructor(window_size)
    p_mask = persistent_mask_constructor(num_persistent_tokens)
    swa_p_mask = swa_persistent_mask_constructor(window_size, num_persistent_tokens)
    inverse_swa_mask = inverse_sliding_window_mask_constructor(window_size)
    inverse_p_mask = inverse_persistent_mask_constructor(num_persistent_tokens)
    top_k_mask = top_k_mask_constructor(window_size, num_persistent_tokens)
    score_mod =  tanh_softcap_constructor(tanh_softcap)

    visualize_attention_scores(
        query,
        key,
        score_mod=None,
        mask_mod=top_k_mask,
        device=device,
        name=f"topk_swa_{window_size}_persistent_{num_persistent_tokens}_mask"
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)