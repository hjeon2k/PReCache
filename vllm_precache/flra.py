"""Flash-LoRA-Attention in vLLM's attention impl, grouped by adapter, plus the LR cache scatter."""
from types import SimpleNamespace

import torch

from . import lrwrite

_method = None
_n_adapters = 0
_rank = 0
_bound = {}                  # attention layer name -> (kv tensor, qkv LoRA layer)
_lr_b = {}                   # (layer name, adapter) -> [n_kv, r, head_size]
_plan = (None, None)         # (attention metadata of the current step, its plan)


def configure(method, n_adapters, rank):
    global _method, _n_adapters, _rank
    _method, _n_adapters, _rank = method, n_adapters, rank


def bind(runner):
    from vllm.lora.layers.column_parallel_linear import MergedQKVParallelLinearWithLoRA

    names = list(runner.kv_cache_config.kv_cache_groups[0].layer_names)
    kv_by_name = dict(zip(names, runner.kv_caches))
    _bound.clear()
    _lr_b.clear()
    for name, mod in runner.model.named_modules():
        if isinstance(mod, MergedQKVParallelLinearWithLoRA):
            attn = name.replace(".qkv_proj", ".attn")
            _bound[attn] = (kv_by_name[attn], mod)


def lr_b(layer_name, lora_layer, adapter, n_kv, head_size, dtype):
    key = (layer_name, adapter)
    if key not in _lr_b:
        if adapter < 0:
            _lr_b[key] = torch.zeros(n_kv, _rank, head_size, dtype=dtype, device="cuda")
        else:
            # vLLM folds the alpha/r scaling into B at load time
            b = lrwrite.saved_b(lora_layer.lora_a_stacked[2].data_ptr())[adapter, 0]   # [n_kv*d, r]
            _lr_b[key] = b.T.reshape(_rank, n_kv, head_size).transpose(0, 1).contiguous().to(dtype)
    return _lr_b[key]


def make_plan(md, token_adapter, bs):
    n = md.num_actual_tokens
    dev = token_adapter.device
    r = torch.arange(_rank, device=dev)

    # LR writes: PreLRShared every adapter; ReBaseShared the active one, or every one in the adapter-free pass
    own = torch.nn.functional.one_hot(token_adapter.clamp(min=0).long(), _n_adapters).bool()
    every = torch.ones_like(own)
    writers = every if _method == "prelrshared" else torch.where((token_adapter < 0).unsqueeze(-1), every, own)
    slots = md.slot_mapping[:n]
    w_tok, w_ad = (writers & (slots >= 0).unsqueeze(-1)).nonzero(as_tuple=True)
    w_slot = slots[w_tok]
    plan = SimpleNamespace(w_tok=w_tok, w_ad=w_ad, w_blk=(w_slot // bs).unsqueeze(-1),
                           w_off=(w_slot % bs).unsqueeze(-1), w_cols=w_ad.unsqueeze(-1) * _rank + r,
                           groups=[])

    # attention: one kernel call per adapter group (adapter-free group: lr_B = 0)
    cu_q = md.query_start_loc
    nseq = int(cu_q.shape[0]) - 1
    q_len = cu_q[1:nseq + 1] - cu_q[:nseq]
    seq_adapter = token_adapter[cu_q[:nseq].long()]
    for adapter in torch.unique(seq_adapter).tolist():
        seqs = (seq_adapter == adapter).nonzero(as_tuple=True)[0]
        ql = q_len[seqs]
        sk = md.seq_lens[seqs].to(torch.int32)
        bt = md.block_table[seqs].to(torch.int32)
        tok = torch.repeat_interleave(cu_q[seqs].long(), ql) + (
            torch.arange(int(ql.sum()), device=dev) - torch.repeat_interleave(torch.cumsum(ql, 0) - ql, ql))
        g = SimpleNamespace(adapter=adapter, lo=max(adapter, 0) * _rank, tok=tok, sk=sk, bt=bt,
                            nseq=len(seqs), max_q=int(ql.max()), max_k=int(sk.max()))
        if g.max_q == 1:
            nblk = (g.max_k + bs - 1) // bs
            g.blocks = bt[:, :nblk].long()
        else:
            pos = torch.arange(g.max_k, device=dev).unsqueeze(0).expand(g.nseq, g.max_k)
            keep = pos < sk.unsqueeze(1)
            g.blk = bt.long().gather(1, (pos // bs).clamp(max=bt.shape[1] - 1))[keep]
            g.off = (pos % bs)[keep]
            g.cu_k = torch.zeros(g.nseq + 1, dtype=torch.int32, device=dev)
            torch.cumsum(sk, 0, out=g.cu_k[1:])
            g.cu_q = torch.zeros(g.nseq + 1, dtype=torch.int32, device=dev)
            torch.cumsum(ql, 0, out=g.cu_q[1:])
        plan.groups.append(g)
    return plan


def rebase_active(r_ids, b_ids, n_tokens, adapter, start=0):
    """Copy agent `adapter`'s LR entries over [start, n) from its blocks onto the published ones."""
    if adapter < 0 or not r_ids or not b_ids:
        return
    bs = lrwrite.block_size()
    n = min(n_tokens, len(r_ids) * bs, len(b_ids) * bs)
    if n <= start:
        return
    pos = torch.arange(start, n, device="cuda")
    blk, off = pos // bs, pos % bs
    rb = torch.as_tensor(r_ids, device="cuda")[blk]
    bb = torch.as_tensor(b_ids, device="cuda")[blk]
    lo = adapter * _rank
    with torch.inference_mode():          # the KV tensors are inference tensors
        for kv_layer, lora_layer in _bound.values():
            sv = lrwrite.strips(lora_layer.lora_a_stacked[2].data_ptr(), kv_layer)
            sv[bb, off, lo:lo + _rank] = sv[rb, off, lo:lo + _rank]


def make_impl_cls(base):
    import flash_lora_attn_2_cuda as kernel

    class PReCacheFlashAttentionImpl(base):
        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                    output_scale=None, output_block_scale=None):
            global _plan
            md = attn_metadata
            if md is None or not _bound:          # profiling run, before binding
                return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                       output, output_scale, output_block_scale)
            if (output_scale is not None or output_block_scale is not None
                    or getattr(md, "use_cascade", False) or getattr(md, "causal", True) is not True
                    or (self.sliding_window is not None and tuple(self.sliding_window) != (-1, -1))):
                raise NotImplementedError("PReCache FLA: unsupported attention configuration")

            kv_layer, lora_layer = _bound[layer.layer_name]
            a_ptr = lora_layer.lora_a_stacked[2].data_ptr()
            sv = lrwrite.strips(a_ptr, kv_layer)
            if _plan[0] is not md:
                token_adapter = lora_layer.punica_wrapper.token_lora_indices[:md.num_actual_tokens]
                _plan = (md, make_plan(md, token_adapter, sv.shape[1]))
            plan = _plan[1]
            lrwrite.scatter(a_ptr, kv_layer, plan)

            kc, vc = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
            for g in plan.groups:
                b = lr_b(layer.layer_name, lora_layer, g.adapter, self.num_kv_heads, self.head_size, query.dtype)
                if g.max_q == 1:
                    lr_v = sv[g.blocks, :, g.lo:g.lo + _rank].reshape(g.nseq, -1, _rank).to(query.dtype)
                    out = kernel.fwd_kvcache(
                        query[g.tok].unsqueeze(1), kc, vc, None, None, lr_v, b, g.sk,
                        None, None, None, None, g.bt, None, None, self.scale, True, -1, -1, 0.0, False, 0)[0]
                    output[g.tok] = out.view(g.nseq, self.num_heads, self.head_size)
                else:
                    # varlen reads lr_v dense [total_k, r], sequence-major
                    lr_v = sv[g.blk, g.off, g.lo:g.lo + _rank].to(query.dtype)
                    out = torch.empty_like(output[g.tok])
                    kernel.varlen_fwd(query[g.tok], kc, vc, lr_v, b, out, g.cu_q, g.cu_k, g.sk, None, g.bt,
                                      None, g.max_q, g.max_k, 0.0, self.scale, False, True, -1, -1,
                                      0.0, False, None)
                    output[g.tok] = out
            return output

    return PReCacheFlashAttentionImpl
