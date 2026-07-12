"""3B-GSM8K variance investigation: is the high clean spread from bad-seed outliers
(undertrained / collapsed), and is the noise pattern symmetric once clean seeds fill?
For each run: final acc, best acc (max val), best-final gap, and a collapse flag
(final << best => climbed then degraded/crashed). Usage: python jobscripts/probe_3b.py
"""
import os, statistics as st
from collections import defaultdict
import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "rm4411-princeton-university")
api = wandb.Api()
runs = api.runs(f"{ENTITY}/RLVR")

cells = defaultdict(list)
for r in runs:
    if not r.name.startswith("p3B-gsm8k"):
        continue
    em = r.config.get("env", {}).get("math", {})
    fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    if fp is None or acc is None:
        continue
    try:
        h = r.history(keys=["validation/accuracy"], samples=60, pandas=False)
        vals = [row.get("validation/accuracy") for row in h if row.get("validation/accuracy") is not None]
    except Exception:
        vals = [acc]
    best = max(vals) if vals else acc
    nsteps = len(vals)
    cells[(round(float(fp), 2), round(float(fn), 2))].append(
        dict(name=r.name, acc=float(acc), best=best, gap=best - float(acc),
             nsteps=nsteps, state=r.state))

print(f"{'='*72}\n3B-GSM8K per-run (final / best / collapse) — {sum(len(v) for v in cells.values())} runs\n{'='*72}")
for cell in sorted(cells):
    lst = cells[cell]
    accs = [x["acc"] for x in lst]
    print(f"\ncell fp={cell[0]} fn={cell[1]}:  final acc={st.mean(accs):.3f}±{st.pstdev(accs) if len(accs)>1 else 0:.3f} (n={len(accs)})")
    for x in sorted(lst, key=lambda z: z["acc"]):
        flag = ""
        if x["gap"] > 0.05: flag += " CLIMB-THEN-DROP"
        if x["nsteps"] < 8: flag += f" FEW-STEPS({x['nsteps']})"
        if x["acc"] < 0.45: flag += " LOW"
        print(f"    {x['name'][-12:]:>12}  final={x['acc']:.3f}  best={x['best']:.3f}  gap={x['gap']:+.3f}  steps={x['nsteps']} {x['state']}{flag}")

# symmetric check with clean (non-collapsed) seeds
print(f"\n{'-'*72}\nasymmetry (drop CLIMB-THEN-DROP outliers, gap>0.05):")
def clean_mean(cell):
    lst = [x for x in cells.get(cell, []) if x["gap"] <= 0.05]
    if not lst: return None, 0
    return st.mean([x["acc"] for x in lst]), len(lst)
for a in (0.3,):
    fp = clean_mean((a, 0.0)); fn = clean_mean((0.0, a)); cl = clean_mean((0.0, 0.0))
    print(f"  clean(cleaned)={cl}  FP({a},0)={fp}  FN(0,{a})={fn}")
    if fp[0] and fn[0]:
        print(f"  FP-FN diff = {fp[0]-fn[0]:+.3f}  ({'symmetric' if abs(fp[0]-fn[0])<0.03 else 'ASYMMETRIC'})")
