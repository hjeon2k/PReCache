"""LR cache write: X @ A_v for every adapter (a traced op) and base-only V in the KV cache."""
import torch

from . import backend

_n_adapters = 0
_rank = 0
_slab = None                 # [layers, n_adapters, max_tokens, rank]
_row_by_aptr = {}            # A_v data_ptr -> row in the slab
_saved_b = {}                # A_v data_ptr -> real B_v stack
_views = {}                  # A_v data_ptr -> strip view
_spec = None
_num_blocks = 0


def allocate(n_adapters, rank, max_tokens, max_layers=64):
    """Before the engine is built: tensors cannot be created inside a captured forward."""
    global _n_adapters, _rank, _slab
    _n_adapters, _rank = n_adapters, rank
    _slab = torch.zeros(max_layers, n_adapters, max_tokens, rank, dtype=torch.float16, device="cuda")


def _stash(slab: torch.Tensor, x: torch.Tensor, a_v: torch.Tensor) -> None:
    row = _row_by_aptr.setdefault(a_v.data_ptr(), len(_row_by_aptr))
    flat = x.reshape(-1, x.shape[-1])
    n = min(flat.shape[0], slab.shape[2])
    a = min(slab.shape[1], a_v.shape[0])
    slab[row, :a, :n] = torch.einsum("td,ard->atr", flat[:n].to(a_v.dtype), a_v[:a, 0]).to(slab.dtype)


def _stash_fake(slab: torch.Tensor, x: torch.Tensor, a_v: torch.Tensor) -> None:
    return


def patch_layers():
    """Before the engine is built, so the op is traced into the graph."""
    from vllm.lora.layers.column_parallel_linear import MergedQKVParallelLinearWithLoRA
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "precache_stash_av"):
        direct_register_custom_op(op_name="precache_stash_av", op_func=_stash,
                                  mutates_args=["slab"], fake_impl=_stash_fake)
    orig = MergedQKVParallelLinearWithLoRA.forward

    def forward(self, x, *a, **kw):
        out = orig(self, x, *a, **kw)
        if len(self.lora_a_stacked) >= 3:
            torch.ops.vllm.precache_stash_av(_slab, x, self.lora_a_stacked[2])
        return out

    MergedQKVParallelLinearWithLoRA.forward = forward


def bind(spec, num_blocks):
    global _spec, _num_blocks
    _spec, _num_blocks = spec, num_blocks
    _views.clear()


def split_v(runner):
    """Zero B_v so the KV cache receives base V. Every adapter must already be resident."""
    from vllm.lora.layers.column_parallel_linear import MergedQKVParallelLinearWithLoRA

    for _, mod in runner.model.named_modules():
        if isinstance(mod, MergedQKVParallelLinearWithLoRA) and len(mod.lora_b_stacked) >= 3:
            key = mod.lora_a_stacked[2].data_ptr()
            _saved_b.setdefault(key, mod.lora_b_stacked[2].detach().clone())
            mod.lora_b_stacked[2].zero_()


def block_size():
    return _spec.block_size


def saved_b(a_v_ptr):
    return _saved_b[a_v_ptr]


def strips(a_v_ptr, kv_layer):
    sv = _views.get(a_v_ptr)
    if sv is None:
        sv = _views[a_v_ptr] = backend.strip_view(kv_layer, _spec, _num_blocks)
    return sv


def scatter(a_v_ptr, kv_layer, plan):
    """Write this step's X @ A_v into the strips, at the (slot, adapter) pairs in `plan`."""
    buf = _slab[_row_by_aptr[a_v_ptr]]                       # [n_adapters, max_tokens, r]
    sv = strips(a_v_ptr, kv_layer)
    sv[plan.w_blk, plan.w_off, plan.w_cols] = buf[plan.w_ad, plan.w_tok].to(sv.dtype)
