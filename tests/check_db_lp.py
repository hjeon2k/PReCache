"""ReBaseShared DB vs LP on identical tokens: adapted logits per step, final base K/V and LR caches.

    python tests/check_db_lp.py adapters/llama-3.1-8b/hotpotqa
"""
import os
import sys

import torch
from transformers.cache_utils import DynamicCache, LRCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import single_stream as B  # noqa: E402

model = B.load(sys.argv[1], "rebaseshared")
torch.manual_seed(0)
V = model.config.vocab_size
turns = [("plan", 300, 6), ("action", 40, 5), ("plan", 120, 4), ("reflect", 30, 3)]
toks = [(torch.randint(0, V, (1, p), device="cuda"), torch.randint(0, V, (1, d), device="cuda")) for _, p, d in turns]


@torch.no_grad()
def run(db):
    lr, kv, logits = {a: LRCache() for a in B.AGENTS}, DynamicCache(), []
    for (agent, _, _), (ids, dec) in zip(turns, toks):
        model.set_adapter(agent)
        start = kv.get_seq_length()
        if db:
            view, kv = B.double_view(kv), None
            kw = {"past_key_values": view, "past_lr_caches": lr, "precache_double_batch": True,
                  "adapter_names": [agent, "__base__"]}
        else:
            kw = {"past_key_values": kv, "past_lr_caches": lr}
        n = 2 if db else 1
        logits.append(B.prefill(model, ids.expand(n, -1), start, **kw).logits[:1, -1].float())
        for i in range(dec.shape[1]):
            logits.append(model(input_ids=dec[:, i:i + 1].expand(n, -1), use_cache=True, **kw).logits[:1, -1].float())
        if db:
            kv = B.neutral_row(view)
        else:
            B.adapter_free(model, kv, lr, torch.cat([ids, dec], 1), start)
    return kv, lr, logits


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


kv_lp, lr_lp, lg_lp = run(db=False)
kv_db, lr_db, lg_db = run(db=True)
L = range(len(kv_lp.key_cache))
print(f"adapted logits      {max(rel(a, b) for a, b in zip(lg_db, lg_lp)):.1e}   "
      f"argmax agree {sum(int(a.argmax() == b.argmax()) for a, b in zip(lg_db, lg_lp))}/{len(lg_lp)}")
print(f"base K / V          {max(rel(kv_db.key_cache[l], kv_lp.key_cache[l]) for l in L):.1e} / "
      f"{max(rel(kv_db.value_cache[l], kv_lp.value_cache[l]) for l in L):.1e}")
for a in B.AGENTS:
    print(f"LR {a:<16} {max(rel(lr_db[a].value_cache[l], lr_lp[a].value_cache[l]) for l in L):.1e}")
