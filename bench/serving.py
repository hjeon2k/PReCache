"""Concurrent serving on vLLM: replay the controlled trajectory at a swept request rate.

    python bench/serving.py --adapters adapters/llama-3.1-8b/hotpotqa --method rebaseshared --qps 0.5,1,2,4,8,16
"""
import argparse
import csv
import json
import os
import random
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # hooks live in the engine core
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"                 # a cached graph would miss the LR cache write

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AGENTS = ("plan", "action", "reflect")


def pct(xs, q):
    ys = sorted(xs)
    k = (len(ys) - 1) * q / 100
    lo = int(k)
    return ys[lo] if lo + 1 >= len(ys) else ys[lo] + (ys[lo + 1] - ys[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="nonshared", choices=["nonshared", "prelrshared", "rebaseshared"])
    ap.add_argument("--adapters", required=True, help="folder with plan/, action/, reflect/")
    ap.add_argument("--trace", default="traces/react_17.3k.csv")
    ap.add_argument("--qps", default="0.5,1,2,4,8,16")
    ap.add_argument("--laps", type=int, default=4, help="trajectories per rate")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.4)   # the paper's A100 setting
    ap.add_argument("--max-num-batched-tokens", type=int, default=None, help="default: vLLM's")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind

    from vllm_precache import backend, cache, flra, lrwrite

    rows = [(r["agent"], int(r["prefill_len"]), int(r["decode_len"])) for r in csv.DictReader(open(args.trace))]
    ends, total = [], 0
    for _, prefill, decode in rows:
        ends.append(total + prefill)          # the turn's prompt is the trajectory up to here
        total += prefill + decode
    cfg = json.load(open(os.path.join(args.adapters, AGENTS[0], "adapter_config.json")))
    rank = cfg["r"]

    shared = args.method != "nonshared"
    stash = args.max_num_batched_tokens or 32768
    if shared:
        backend.configure(len(AGENTS), rank)
        backend.install()
        lrwrite.allocate(len(AGENTS), rank, stash)
        lrwrite.patch_layers()
    cache.install(args.method)

    llm = LLM(model=cfg["base_model_name_or_path"], enable_lora=True, max_loras=len(AGENTS),
              max_lora_rank=rank, max_model_len=total + 256, block_size=256,
              enable_prefix_caching=True, disable_cascade_attn=True,
              **({"max_num_batched_tokens": args.max_num_batched_tokens} if args.max_num_batched_tokens else {}),
              gpu_memory_utilization=args.gpu_memory_utilization,
              compilation_config={"cudagraph_mode": "PIECEWISE"}, disable_log_stats=True)
    loras = {a: LoRARequest(a, i + 1, os.path.abspath(os.path.join(args.adapters, a))) for i, a in enumerate(AGENTS)}
    rng = random.Random(0)
    warm = TokensPrompt(prompt_token_ids=[rng.randrange(1000, 30000) for _ in range(64)])
    for lo in loras.values():                 # every adapter resident before B_v is moved
        llm.generate([warm], SamplingParams(max_tokens=1), lora_request=lo, use_tqdm=False)

    engine = llm.llm_engine
    assert not shared or engine.vllm_config.scheduler_config.max_num_batched_tokens <= stash, \
        "raise --max-num-batched-tokens"
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    spec = next(iter(core.model_executor.get_kv_cache_specs()[0].values()))
    num_blocks = engine.vllm_config.cache_config.num_gpu_blocks
    if shared:
        lrwrite.bind(spec, num_blocks)
        lrwrite.split_v(runner)
        flra.configure(args.method, len(AGENTS), rank)
        flra.bind(runner)

    def replay(qps, laps):
        engine.reset_prefix_cache()
        cache.reset()
        streams = [[rng.randrange(1000, 30000) for _ in range(total)] for _ in range(laps)]
        free_sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True,
                                 output_kind=RequestOutputKind.DELTA)
        sched = []
        for k in range(laps * len(rows)):
            lap, turn = divmod(k, len(rows))
            agent, _, decode = rows[turn]
            prompt = TokensPrompt(prompt_token_ids=streams[lap][:ends[turn]])
            sp = SamplingParams(temperature=0.0, max_tokens=decode, ignore_eos=True,
                                output_kind=RequestOutputKind.DELTA)
            sched.append((k / qps, f"c{lap}#r{k}", prompt, sp, loras[agent]))
            if args.method == "rebaseshared":    # DB: adapter-free pass over LP's span (prompt + decoded tokens)
                b_prompt = TokensPrompt(
                    prompt_token_ids=streams[lap][:ends[turn] + decode])
                sched.append((k / qps, f"c{lap}#b{k}", b_prompt, free_sp, None))
        first, done = {}, {}
        t0, nxt = time.perf_counter(), 0
        while nxt < len(sched) or engine.has_unfinished_requests():
            now = time.perf_counter() - t0
            while nxt < len(sched) and sched[nxt][0] <= now:
                _, rid, prompt, sp, lo = sched[nxt]
                engine.add_request(rid, prompt, sp, lora_request=lo)
                nxt += 1
            if not engine.has_unfinished_requests():
                time.sleep(min(0.002, max(0.0, sched[nxt][0] - now)))
                continue
            for o in engine.step():
                t = time.perf_counter() - t0
                if any(c.token_ids for c in o.outputs):
                    first.setdefault(o.request_id, t)
                if o.finished:
                    done[o.request_id] = t
        ttft, turn_latency = [], []
        for k in range(laps * len(rows)):
            lap = k // len(rows)
            arrival, r = k / qps, f"c{lap}#r{k}"
            ttft.append(first[r] - arrival)
            turn_latency.append(max(done[r], done.get(f"c{lap}#b{k}", 0.0)) - arrival)
        latency = sum(turn_latency) / laps       # one trajectory's turns, end to end
        return ttft, latency

    rates = [float(q) for q in args.qps.split(",")]
    for q in (min(rates), max(rates)):        # warmup: compile the batched LoRA kernel shapes too
        replay(q, 1)
    kv_bytes_per_block = spec.page_size_bytes * engine.vllm_config.model_config.get_num_layers(engine.vllm_config.parallel_config)
    print(f"method {args.method}  trace {os.path.basename(args.trace)}  trajectory tokens {total}")
    print(f"{'QPS':>6}{'p50 TTFT (s)':>14}{'p90 TTFT (s)':>14}{'latency (s)':>13}"
          f"{'TP (tok/s)':>12}{'peak KV (GiB)':>15}")
    for qps in rates:
        ttft, latency = replay(qps, args.laps)
        print(f"{qps:>6g}{pct(ttft, 50):>14.2f}{pct(ttft, 90):>14.2f}{latency:>13.2f}"
              f"{total / latency:>12.0f}{cache.peak['blocks'] * kv_bytes_per_block / 2**30:>15.2f}", flush=True)


if __name__ == "__main__":
    main()
