"""Check each method's caches against its definition, through the patched FastChat generation.

    python tests/check_methods.py adapters/llama-3.1-8b/hotpotqa
"""
import json
import os
import sys

import torch
from fastchat.serve.inference import generate_stream
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, LRCache

AGENTS = ("plan", "action", "reflect")
TEXT = ["The quick brown fox jumps over the lazy dog. Question: what did the fox do? Answer:",
        " Next, the dog woke up and said something to the fox. The dog said:",
        " Then the farmer came out of the house and asked:"]
AGENT = ["plan", "action", "plan"]

adapters = sys.argv[1]
base = json.load(open(os.path.join(adapters, "plan", "adapter_config.json")))["base_model_name_or_path"]
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                             attn_implementation="flash_attention_2").cuda()
model = PeftModel.from_pretrained(model, os.path.join(adapters, "plan"), adapter_name="plan")
for name in AGENTS[1:]:
    model.load_adapter(os.path.join(adapters, name), adapter_name=name)
model.eval()
cfg = model.config
layers = model.base_model.model.model.layers
n_kv, head_dim = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads

forward = model.forward


def generate(agent, prompt):
    model.set_adapter(agent)
    text = ""
    for out in generate_stream(model, tok, {"prompt": prompt, "temperature": 0.0, "max_new_tokens": 6},
                               "cuda", 8192):
        text = out["text"]
    return text


def fresh(method):
    model.config.precache_method = method
    model.precache = (DynamicCache(), {name: LRCache() for name in AGENTS}, [])


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run(method):
    """Three turns, each prompt extending the previous one and its output; returns each turn's span."""
    fresh(method)
    held = model.precache[2]
    starts, prompt = [], ""
    for agent, text in zip(AGENT, TEXT):
        prompt += text
        before = list(held)
        prompt += generate(agent, prompt)
        starts.append(common_prefix(before, held))
    ends = starts[1:] + [len(held)]
    return list(zip(starts, ends))


def reference(ids, adapter, past=None, start=0):
    """Normalised layer inputs and K/V of a plain forward (adapter None: adapter-free)."""
    hs = []
    hooks = [l.input_layernorm.register_forward_hook(lambda m, i, o: hs.append(o[0].float())) for l in layers]
    kw = {"past_key_values": past, "cache_position": torch.arange(start, start + len(ids), device="cuda")}
    try:
        if adapter is None:
            with model.disable_adapter():
                out = forward(input_ids=ids[None], use_cache=True, **kw)
        else:
            model.set_adapter(adapter)
            out = forward(input_ids=ids[None], use_cache=True, **kw)
    finally:
        for h in hooks:
            h.remove()
    return hs, out.past_key_values


def lr(adapter, l, h):
    return layers[l].self_attn.v_proj.lora_A[adapter](h)


def err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def worst(pairs):
    return max(err(a, b) for a, b in pairs)


L = range(len(layers))
with torch.no_grad():
    spans = run("rebaseshared")
    kv, lrc, held = model.precache
    full = torch.tensor(held, device="cuda")                    # the tokens the caches hold
    hs, ref = reference(full, None)
    print(f"rebaseshared  base K/V == adapter-free forward        "
          f"{worst([(kv.key_cache[l], ref.key_cache[l]) for l in L] + [(kv.value_cache[l], ref.value_cache[l]) for l in L]):.1e}")
    print(f"rebaseshared  reflect LR == X_neut A                  "
          f"{worst([(lrc['reflect'].value_cache[l][0], lr('reflect', l, hs[l])) for l in L]):.1e}")
    s, e = spans[1]
    print(f"rebaseshared  plan LR over action's turn == X_neut A  "
          f"{worst([(lrc['plan'].value_cache[l][0, s:e], lr('plan', l, hs[l][s:e])) for l in L]):.1e}")
    # action's own turn: adapted pass over the neutral prefix, with action's LR term on V
    hs_pre, past = reference(full[:s], None)
    for l in L:
        v = layers[l].self_attn.v_proj
        d = v.lora_B["action"](v.lora_A["action"](hs_pre[l])) * v.scaling["action"]
        past.value_cache[l] += d.view(1, s, n_kv, head_dim).transpose(1, 2).to(past.value_cache[l].dtype)
    hs_a, _ = reference(full[s:e], "action", past, s)
    print(f"rebaseshared  action LR over its turn == adapted pass "
          f"{worst([(lrc['action'].value_cache[l][0, s:e], lr('action', l, hs_a[l])) for l in L]):.1e}")

    spans = run("prelrshared")
    kv, lrc, held = model.precache
    s, e = spans[0]
    full = torch.tensor(held, device="cuda")
    hs, ref = reference(full[s:e], "plan")
    print(f"prelrshared   all LR over plan's turn == X_plan A     "
          f"{worst([(lrc[a].value_cache[l][0, s:e], lr(a, l, hs[l])) for a in AGENTS for l in L]):.1e}")
    print(f"prelrshared   base K over plan's turn == plan forward "
          f"{worst([(kv.key_cache[l][:, :, s:e], ref.key_cache[l]) for l in L]):.1e}")

    fresh("prelrshared")
    prompt = TEXT[0]
    generate("plan", prompt + generate("plan", prompt))
    kv, lrc, held = model.precache

    # a prompt that disagrees with the cache: caches are cut back to the common prefix
    ids = tok(TEXT[0] + " A different continuation entirely.").input_ids
    generate("plan", TEXT[0] + " A different continuation entirely.")
    print(f"divergent     cache holds the new prompt: {held[:len(ids)] == ids}; KV length == "
          f"held tokens: {kv.get_seq_length() == len(held)}; every LR cache == KV: "
          f"{all(c.get_seq_length() == kv.get_seq_length() for c in lrc.values())}")
