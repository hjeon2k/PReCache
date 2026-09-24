"""Exact-match accuracy (%) per level and on average over all trials (plus F1 for HotpotQA)."""
import json
import os
import sys

root = sys.argv[1]
rows = []
for level in sorted(os.listdir(root)):
    runs = sorted(f for f in os.listdir(os.path.join(root, level)) if f.endswith(".jsonl"))
    items = [json.loads(line) for f in runs for line in open(os.path.join(root, level, f)) if line.strip()]
    if not items:
        continue
    acc = 100 * sum(bool(x["correct"]) for x in items) / len(items)
    f1 = 100 * sum(float(x["reward"]) for x in items) / len(items)
    rows.append((level, len(runs), len(items), acc, f1))
hotpotqa = {r[0] for r in rows} <= {"easy", "medium", "hard"}
for level, trials, n, acc, f1 in rows + [("avg", "", "", sum(r[3] for r in rows) / max(1, len(rows)),
                                          sum(r[4] for r in rows) / max(1, len(rows)))]:
    print(f"{level:>8}  trials {trials:>3}  questions {n:>5}  acc {acc:6.2f}" + (f"  f1 {f1:6.2f}" if hotpotqa else ""))
