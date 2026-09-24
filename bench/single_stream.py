"""Single-stream efficiency: replay a controlled trajectory one agent turn at a time.

    python bench/single_stream.py --adapters adapters/llama-3.1-8b/hotpotqa --method rebaseshared --schedule lp
"""
import argparse
import csv
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, LRCache

AGENTS = ("plan", "action", "reflect")


def load(adapters, method):
    base = json.load(open(os.path.join(adapters, AGENTS[0], "adapter_config.json")))["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
    model = PeftModel.from_pretrained(model, os.path.join(adapters, AGENTS[0]), adapter_name=AGENTS[0])
    for name in AGENTS[1:]:
        model.load_adapter(os.path.join(adapters, name), adapter_name=name)
    if method != "nonshared":
        model.config.precache_method = method
    return model.eval()


def sync():
    torch.cuda.synchronize()


def prefill(model, ids, start, **kw):
    return model(input_ids=ids, use_cache=True, logits_to_keep=1,
                 cache_position=torch.arange(start, start + ids.shape[1], device="cuda"), **kw)


def adapter_free(model, kv, lr, ids, start):
    # ReBaseShared LP: rewrite the base K/V of `ids` without adapters and extend the other agents' LR caches
    kv.crop(start)
    with model.disable_adapter():
        prefill(model, ids, start, past_key_values=kv, past_lr_caches=lr)


def double_view(kv):
    # ReBaseShared DB: row 0 is the adapted view, row 1 the adapter-free one, both over the neutral prefix
    view = DynamicCache()
    view.key_cache = [k.expand(2, -1, -1, -1) for k in kv.key_cache]
    view.value_cache = [v.expand(2, -1, -1, -1) for v in kv.value_cache]
    view._seen_tokens = kv._seen_tokens
    return view


def neutral_row(view):
    kv = DynamicCache()
    for i in range(len(view.key_cache)):     # layer by layer, releasing the two-row view as it goes
        kv.key_cache.append(view.key_cache[i][1:].contiguous())
        kv.value_cache.append(view.value_cache[i][1:].contiguous())
        view.key_cache[i] = view.value_cache[i] = None
    kv._seen_tokens = view._seen_tokens
    return kv


@torch.no_grad()
def run_trajectory(model, rows, method, schedule="lp"):
    vocab = model.config.vocab_size
    shared = method != "nonshared"
    db = method == "rebaseshared" and schedule == "db"
    lp = method == "rebaseshared" and not db
    lr = {name: LRCache() for name in AGENTS} if shared else None
    caches = {name: DynamicCache() for name in AGENTS}     # NonShared: one KV cache per agent
    kv, committed, ttft = DynamicCache(), 0, 0.0
    torch.cuda.reset_peak_memory_stats()
    sync()
    t_start = time.perf_counter()
    for agent, prefill_len, decode_len in rows:
        model.set_adapter(agent)
        if not shared:
            kv = caches[agent]
        start = kv.get_seq_length()                  # NonShared re-prefills what others added since
        turn_start, end = committed, committed + prefill_len
        ids = torch.randint(0, vocab, (1, end - start), device="cuda")
        if db:                                       # one forward serves both paths, prefill and decoding
            view, kv = double_view(kv), None         # the view alone keeps the prefix alive
            kw = {"past_key_values": view, "past_lr_caches": lr, "precache_double_batch": True,
                  "adapter_names": [agent, "__base__"]}
        else:
            kw = {"past_key_values": kv, **({"past_lr_caches": lr} if shared else {})}
        rows_in = 2 if db else 1
        fed = [ids]
        sync()
        t0 = time.perf_counter()
        token = prefill(model, ids.expand(rows_in, -1), start, **kw).logits[:1, -1:].argmax(-1)
        sync()
        ttft += time.perf_counter() - t0
        for _ in range(decode_len):
            fed.append(token)
            out = model(input_ids=token.expand(rows_in, -1), use_cache=True, **kw)
            token = out.logits[:1, -1:].argmax(-1)
        if db:
            kv = neutral_row(view)
        if lp:                                       # lazy prefill of the turn's own tokens, outside TTFT
            adapter_free(model, kv, lr, torch.cat(fed, dim=1), turn_start)
        committed = kv.get_seq_length()
    sync()
    latency = time.perf_counter() - t_start
    return ttft, latency, committed, torch.cuda.max_memory_allocated()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="nonshared", choices=["nonshared", "prelrshared", "rebaseshared"])
    ap.add_argument("--adapters", required=True, help="folder with plan/, action/, reflect/")
    ap.add_argument("--trace", default="traces/react_17.3k.csv")
    ap.add_argument("--schedule", default="lp", choices=["lp", "db"], help="ReBaseShared only")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    rows = [(r["agent"], int(r["prefill_len"]), int(r["decode_len"])) for r in csv.DictReader(open(args.trace))]
    model = load(args.adapters, args.method)
    run_trajectory(model, rows, args.method, args.schedule)          # warmup
    runs = []
    for _ in range(args.repeats):
        runs.append(run_trajectory(model, rows, args.method, args.schedule))
        torch.cuda.empty_cache()
    ttft, latency = (sum(r[i] for r in runs) / len(runs) for i in (0, 1))
    name = args.method + (f"-{args.schedule}" if args.method == "rebaseshared" else "")
    print(f"method {name}  trace {os.path.basename(args.trace)}  repeats {len(runs)}")
    print(f"  TTFT (s)               {ttft:.3f}")
    print(f"  total latency (s)      {latency:.3f}")
    print(f"  trajectory tokens      {runs[0][2]}")
    print(f"  peak memory (GiB)      {max(r[3] for r in runs) / 2**30:.2f}")


if __name__ == "__main__":
    main()
