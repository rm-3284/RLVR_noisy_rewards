"""Same rigorous usable count (dedup by max _step, require >=10 real validation evals),
applied to GSM8K grids — esp. the OLMo-1B cube's r=128 layer."""
import re
from collections import defaultdict
import wandb
api = wandb.Api()
PROJ = "rm4411-princeton-university/RLVR"
MIN_EVALS = 10
GRIDS = [
    ("OLMo-1B/GSM8K (cube)", r"^aiOLMOgsm-1B-r", 240),
    ("Qwen-3B/GSM8K",        r"^ai3Bgsm-3B-r",   None),
]
def rollout_of(n):
    m = re.search(r"-r(\d+)-", n); return int(m.group(1)) if m else 0
for label, rgx, target in GRIDS:
    best = {}
    for r in api.runs(PROJ, filters={"display_name": {"$regex": rgx}}, per_page=500):
        if r.state != "finished": continue
        s = r.summary.get("_step", 0) or 0
        if r.name not in best or s > best[r.name][0]: best[r.name] = (s, r)
    per = defaultdict(lambda: [0, 0]); thin = 0
    for name, (s, r) in best.items():
        roll = rollout_of(name); per[roll][1] += 1
        try:
            h = r.history(keys=["validation/accuracy"], pandas=False)
            n = sum(1 for row in h if row.get("validation/accuracy") is not None)
        except Exception: n = 0
        if n >= MIN_EVALS: per[roll][0] += 1
        else: thin += 1
    u = sum(v[0] for v in per.values()); c = sum(v[1] for v in per.values())
    tgt = f"target {target}" if target else "no fixed target"
    print(f"\n{label}: {u} USABLE of {c} distinct cells  [{tgt}]  | thin/dropped: {thin}")
    for roll in sorted(per):
        uu, tt = per[roll]; print(f"    r{roll:<4} {uu:>3}/{tt:<3} usable")
print("\nDONE")
