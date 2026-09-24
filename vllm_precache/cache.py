"""Prefix-cache rules in the engine core: adapter-free block keys, and DB's private adapted view."""
import re

_allocated = set()
peak = {"blocks": 0}
_neutral = {}          # adapter-free request -> (blocks, prompt length)
_spanned = {}          # conversation -> where the previous turn's span ended


def _external(request):
    # the engine appends "-<8 hex>" to the id it was given
    return re.sub(r"-[0-9a-f]{8}$", "", request.request_id)


def install(method):
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

    if method != "nonshared":               # agents share base blocks
        kv_cache_utils._gen_lora_extra_hash_keys = lambda request: []

    orig_allocate = KVCacheManager.allocate_slots

    def allocate_slots(self, request, *args, **kwargs):
        out = orig_allocate(self, request, *args, **kwargs)
        if out is not None:
            _allocated.add(_external(request))
            pool = self.block_pool
            peak["blocks"] = max(peak["blocks"], pool.num_gpu_blocks - pool.get_num_free_blocks())
        return out

    KVCacheManager.allocate_slots = allocate_slots

    if method != "rebaseshared":
        return
    orig_cache = SingleTypeKVCacheManager.cache_blocks

    # DB: the adapted request (#r) never publishes; the adapter-free request (#b) only after #r is allocated
    def cache_blocks(self, request, num_tokens, *args, **kwargs):
        rid = _external(request)
        if "#r" in rid:
            return
        if "#b" in rid and rid.replace("#b", "#r") not in _allocated:
            return          # not advanced: published by a later call
        return orig_cache(self, request, num_tokens, *args, **kwargs)

    SingleTypeKVCacheManager.cache_blocks = cache_blocks

    orig_free = KVCacheManager.free

    # DB: agent j's LR cache for its own turn is the adapted one, so #r's column moves onto #b's published blocks before #r's are freed
    def free(self, request, *args, **kwargs):
        rid = _external(request)
        try:
            ids = self.get_block_ids(request.request_id)[0]
        except Exception:
            ids = None
        if ids and "#b" in rid:
            _neutral[rid] = (ids, request.num_tokens - request.num_output_tokens)
        elif ids and "#r" in rid:
            from . import flra
            peer = _neutral.pop(rid.replace("#r", "#b"), None)
            lora = getattr(request, "lora_request", None)
            if peer is not None and lora is not None:
                conv = rid.split("#", 1)[0]
                end = min(request.num_tokens, peer[1])
                flra.rebase_active(ids, peer[0], end, lora.lora_int_id - 1,
                                   start=_spanned.get(conv, 0))
                _spanned[conv] = end
        return orig_free(self, request, *args, **kwargs)

    KVCacheManager.free = free


def reset():
    _allocated.clear()
    _neutral.clear()
    _spanned.clear()
    peak["blocks"] = 0
