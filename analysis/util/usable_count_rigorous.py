"""Rigorous usable-cell count: dedup killed+resubmitted runs (keep max _step per name),
then require >=10 real validation evals (actual W&B history), tally per grid x rollout vs target."""
import re
from collections import defaultdict
import wandb
api = wandb.Api()
PROJ = "rm4411-princeton-university/RLVR"
MIN_EVALS = 10

GRIDS = [
    ("Qwen-0.5B/MATH", r"^v4q05-r",   240),
    ("Qwen-1.5B/MATH", r"^v4q15-r",   240),
    ("Qwen-3B/MATH",   r"^v4q3pli-r", 240),
    ("OLMo-1B/MATH-g", r"^v4o1-r",    240),
    ("OLMo-1B/MATH-a", r"^v4o1ail-r", 240),
    ("OLMo-7B/MATH",   r"^v4o7-r",    240),
]

def rollout_of(name):
    m = re.search(r"-r(\d+)-", name)
    return int(m.group(1)) if m else 0

for label, rgx, target in GRIDS:
    # 1) pull all finished runs, dedup by NAME keeping max _step (the fuller of any dup pair)
    best = {}
    for r in api.runs(PROJ, filters={"display_name": {"$regex": rgx}}, per_page=500):
        if r.state != "finished":
            continue
        step = r.summary.get("_step", 0) or 0
        if r.name not in best or step > best[r.name][0]:
            best[r.name] = (step, r)
    # 2) for each deduped cell, count ACTUAL validation evals; usable if >=MIN_EVALS
    per_roll = defaultdict(lambda: [0, 0])  # rollout -> [usable, total_cells]
    thin = 0
    for name, (step, r) in best.items():
        roll = rollout_of(name)
        per_roll[roll][1] += 1
        try:
            h = r.history(keys=["validation/accuracy"], pandas=False)
            n = sum(1 for row in h if row.get("validation/accuracy") is not None)
        except Exception:
            n = 0
        if n >= MIN_EVALS:
            per_roll[roll][0] += 1
        else:
            thin += 1
    usable = sum(v[0] for v in per_roll.values())
    cells  = sum(v[1] for v in per_roll.values())
    print(f"\n{label}: {usable} USABLE (>= {MIN_EVALS} evals) of {cells} distinct cells  [target {target}]  | thin/dropped: {thin}")
    for roll in sorted(per_roll):
        u, t = per_roll[roll]
        print(f"    r{roll:<4} {u:>3}/{t:<3} usable")
print("\nDONE")
