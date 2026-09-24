"""Flash-LoRA-Attention (forward only): softmax(Q K^T * scale) @ (V + lr_v @ lr_B)."""
import torch

import flash_lora_attn_2_cuda as _C

__all__ = ["flash_lora_attn_func"]


def flash_lora_attn_func(q, k, v, lr_v, lr_B, softmax_scale=None, causal=False, window_size=(-1, -1)):
    # q (b, sq, h, d); k, v (b, sk, hk, d); lr_v (b, sk, r), r % 8 == 0; lr_B (hk, r, d); bottom-right causal
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    q, k, v, lr_v, lr_B = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v, lr_v, lr_B)]
    out, _, _, _ = _C.fwd(q, k, v, lr_v, lr_B, None, None, 0.0, softmax_scale, causal,
                          window_size[0], window_size[1], 0.0, False, None)
    return out
