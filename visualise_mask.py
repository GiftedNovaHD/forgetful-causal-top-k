from mask import (sliding_window_mask_constructor,
                  persistent_mask_constructor,
                  swa_persistent_mask_constructor
                 )
import torch

def main(device: str = "cpu"):
    """Visualize the attention scores of sliding window mask mod.

    Args:
        device (str): Device to use for computation. Defaults
    """
    from attn_gym_utils import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 24, 8

    window_size = 4
    num_persistent_tokens = 2

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN + num_persistent_tokens, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    mask = swa_persistent_mask_constructor(window_size, num_persistent_tokens)
    visualize_attention_scores(
        query, key, mask_mod=mask, device=device, name=f"swa_{window_size}_persistent_{num_persistent_tokens}_mask"
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)