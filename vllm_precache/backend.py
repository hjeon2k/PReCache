"""FlashAttention backend whose KV pages carry each token's per-adapter LR values in their tail."""
from dataclasses import replace

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

_n_adapters = 0
_rank = 0


def configure(n_adapters, rank):
    global _n_adapters, _rank
    _n_adapters, _rank = int(n_adapters), int(rank)


def widen(spec):
    if _n_adapters <= 0 or spec.page_size_padded is not None:
        return spec
    size = torch.tensor([], dtype=spec.dtype).element_size()
    tail = spec.block_size * _n_adapters * _rank * size
    return replace(spec, page_size_padded=spec.unpadded_page_size_bytes + tail)


def strip_view(kv_layer, spec, num_blocks):
    """[num_blocks, block_size, n_adapters * rank] over one layer's page tails."""
    size = torch.tensor([], dtype=spec.dtype).element_size()
    width = _n_adapters * _rank
    flat = torch.empty(0, dtype=spec.dtype, device=kv_layer.device)
    flat.set_(kv_layer.untyped_storage())
    return torch.as_strided(flat, size=(num_blocks, spec.block_size, width),
                            stride=(spec.page_size_bytes // size, width, 1),
                            storage_offset=spec.unpadded_page_size_bytes // size)


class PReCacheFlashAttentionBackend(FlashAttentionBackend):          # keeps FLASH_ATTN's name

    @classmethod
    def customize_spec(cls, spec):
        return widen(spec)

    @staticmethod
    def get_impl_cls():
        from . import flra
        return flra.make_impl_cls(FlashAttentionImpl)


def install():
    from vllm.v1.attention import selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.FLASH_ATTN,
                     "vllm_precache.backend.PReCacheFlashAttentionBackend")
    for name in ("_cached_get_attn_backend", "_cached_get_mamba_attn_backend"):
        fn = getattr(selector, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
