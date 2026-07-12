"""OLMo-2-1B GSM8K asymmetry probe: is it FP-worse (predicted, low base rate) or symmetric?
Server-side name filter, summary-only, dedup by max _step, drop truncated. -> logs/OLMo_probe.md read."""
import wandb, statistics as st
from collections import defaultdict
api = wandb.Api()
flt = {"display_name": {"$regex": "^olmo-gsm8k-"}}
best = {}
maxstep = 0
for r in api.runs("rm4411-princeton-university/RLVR", filters=flt, per_page=500):
    em = r.config.get("env", {}).get("math", {}); fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    if fp is None or acc is None: continue
    step = r.summary.get("_step", 0) or 0
    maxstep = max(maxstep, step)
    key = r.name
    if key not in best or step > best[key][0]:
        best[key] = (step, round(float(fp), 2), round(float(fn), 2), float(acc))
# completed = _step >= 80% of the max observed step (OLMo full-run length may differ from Qwen)
thr = 0.8 * maxstep
cells = defaultdict(list); trunc = defaultdict(int)
for step, fp, fn, acc in best.values():
    if step < thr: trunc[(fp, fn)] += 1; continue
    cells[(fp, fn)].append(acc)
print(f"max _step observed = {maxstep}; completed threshold = {thr:.0f}")
print("OLMo-2-1B GSM8K cells (completed, deduped):")
for c in sorted(cells):
    v = cells[c]
    print(f"  fp={c[0]} fn={c[1]}: acc={st.mean(v):.3f}±{st.pstdev(v) if len(v)>1 else 0:.3f} (n={len(v)})  [{trunc.get(c,0)} truncated]")
def cm(c):
    v = cells.get(c, []); return (st.mean(v), len(v)) if v else (None, 0)
cl = cm((0.0, 0.0)); fp = cm((0.3, 0.0)); fn = cm((0.0, 0.3)); mid = cm((0.15, 0.15)); dd = cm((0.3, 0.3))
print(f"\nclean={cl}  FP(0.3,0)={fp}  FN(0,0.3)={fn}  mid(.15,.15)={mid}  (.3,.3)={dd}")
if fp[0] is not None and fn[0] is not None:
    d = fp[0] - fn[0]
    print(f"FP-FN diff = {d:+.3f}  => {'SYMMETRIC' if abs(d)<0.03 else 'FP-WORSE' if d<0 else 'FN-WORSE'}")
if fp[0] is not None and fn[0] is not None and mid[0] is not None:
    sp = max(fp[0], fn[0], mid[0]) - min(fp[0], fn[0], mid[0])
    print(f"margin-collapse m=0.7 spread = {sp:.3f} (holds if ~0.03; breaks if large like MATH-1.5B 0.166)")
print("DONE")
