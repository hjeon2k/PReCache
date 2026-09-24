# PReCache

KV cache sharing for multi-LoRA agent systems: agents share one base KV cache, and each
keeps a rank-r LR cache expanded inside Flash-LoRA-Attention.

| Method | Switch |
|---|---|
| NonShared | (default) |
| PreLRShared | `prelrshared` |
| ReBaseShared (LP / DB) | `rebaseshared` |

## Layout

```
src/flash_lora_attn/   Flash-LoRA-Attention CUDA kernel
patches/               changes to transformers 4.50.0 and FastChat 0.2.35
agents/                plan/action/reflect agents for HotpotQA and ScienceQA
bench/                 single-stream (transformers) and concurrent-serving (vLLM) benchmarks
vllm_precache/         vLLM backend for the LR cache
traces/                controlled 3-agent trajectory (17.3k tokens)
tests/                 cache checks against each method's definition
```

## Install

```bash
# accuracy and single-stream (Python 3.11, CUDA 12.x, sm80+)
python3.11 -m venv .venv && source .venv/bin/activate
TORCH_CUDA_ARCH_LIST=8.0 bash scripts/install.sh

# concurrent serving (Python 3.12; keep it activated when running)
python3.12 -m venv .venv-vllm && source .venv-vllm/bin/activate
TORCH_CUDA_ARCH_LIST=8.0 bash scripts/install_vllm.sh
```

## Adapters

Three PEFT LoRA adapters of one base model (targeting `v_proj`, not `k_proj`; rank a
multiple of 8), one folder per agent:

```
adapters/llama-3.1-8b/hotpotqa/{plan,action,reflect}/{adapter_config.json,adapter_model.safetensors}
```

## Accuracy

```bash
bash scripts/serve.sh adapters/llama-3.1-8b/hotpotqa rebaseshared     # method omitted: NonShared
cd agents
python run_eval.py --task Hotpotqa --task_path benchmark_run/data/hotpotqa --save_path ../results/hotpotqa
python run_eval.py --task Scienceqa --agent_name ZeroshotThink_ScienceQA_run_Agent \
    --task_path benchmark_run/data/scienceqa --save_path ../results/scienceqa
python score.py ../results/hotpotqa
```

`run_eval.py` repeats each benchmark 20 times (`--trials`). Web search is not included:
implement `call_web_search()` in `agents/benchmark_run/utils.py` (the paper uses the Serper API).

## Efficiency

```bash
# single-stream
python bench/single_stream.py --adapters adapters/llama-3.1-8b/hotpotqa --method rebaseshared --schedule lp   # or db

# concurrent serving (vLLM environment)
python bench/serving.py --adapters adapters/llama-3.1-8b/hotpotqa --method rebaseshared --qps 0.5,1,2,4,8,16
```

Omit `--method` for NonShared. ReBaseShared uses LP or DB (`--schedule`) in single-stream
and DB in serving. Both replay `traces/react_17.3k.csv`.
