"""Fast C2 fit: server-side name filter, summary-only, saturating fit with r=64."""
import wandb, re, statistics as st
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit
api = wandb.Api()
flt = {"$or": [{"display_name": {"$regex": "^c2-gsm8k-1.5B"}},
               {"display_name": {"$regex": "^c1-1.5B"}},
               {"display_name": {"$regex": "^c1r8-1.5B"}},
               {"display_name": {"$regex": "^c2b-gsm8k-1.5B"}},
               {"display_name": {"$regex": "^c2c-gsm8k-1.5B"}}]}
NR = re.compile(r"-r(\d+)-s\d+$")
best = {}
for r in api.runs("rm4411-princeton-university/RLVR", filters=flt, per_page=500):
    em = r.config.get("env", {}).get("math", {}); fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    if fp is None or acc is None: continue
    fp, fn = round(float(fp), 2), round(float(fn), 2)
    if (fp, fn) not in [(0.0, 0.0), (0.2, 0.2)]: continue
    m = NR.search(r.name)
    if not m: continue
    step = r.summary.get("_step", 0) or 0
    if r.name not in best or step > best[r.name][0]:
        best[r.name] = (step, fp, fn, int(m.group(1)), float(acc))
arms = defaultdict(lambda: defaultdict(list))
for step, fp, fn, rr, acc in best.values(): arms[(fp, fn)][rr].append(acc)
def rmean(v):
    ok = [x for x in v if x >= 0.10]; return st.mean(ok if ok else v)
def sat(r, A, B, c): return A - B * np.power(r, -c)
res = {}
for arm in [(0.0, 0.0), (0.2, 0.2)]:
    d = arms[arm]; rs = sorted(d); xs = []; ys = []
    print(f"\narm {arm}:")
    for rr in rs:
        v = d[rr]; ncol = len([x for x in v if x < 0.10]); mn = rmean(v)
        print(f"  r={rr:>3}: {mn:.3f} (n={len(v)}{', ' + str(ncol) + ' collapsed' if ncol else ''})")
        xs.append(rr); ys.append(mn)
    if len(xs) >= 4:
        try:
            popt, _ = curve_fit(sat, xs, ys, p0=[max(ys), .4, .5], bounds=([0, 0, .05], [1, 2, 3]), maxfev=40000)
            pred = sat(np.array(xs), *popt); r2 = 1 - np.sum((np.array(ys) - pred) ** 2) / (np.var(ys) * len(ys) + 1e-12)
            print(f"  FIT A={popt[0]:.3f} B={popt[1]:.3f} c={popt[2]:.2f} R2_in={r2:.3f}")
            res[arm] = popt[0]
        except Exception as e:
            print("  fit fail", e)
if len(res) == 2:
    print(f"\nASYMPTOTE gap A_clean-A_noisy = {res[(0.0,0.0)]-res[(0.2,0.2)]:+.3f} (clean {res[(0.0,0.0)]:.3f}, noisy {res[(0.2,0.2)]:.3f})")
cd = arms[(0.0, 0.0)]; nd = arms[(0.2, 0.2)]
print("finite-r gap:", {rr: round(rmean(cd[rr]) - rmean(nd[rr]), 3) for rr in sorted(set(cd) & set(nd))})
print("DONE")
