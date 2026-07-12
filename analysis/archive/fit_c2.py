"""C2 compute axis: acc vs rollouts r, for clean (0,0) and noisy (0.2,0.2).
Pulls c2-gsm8k-* (r 4/16/64) + reuses C1 anchors: c1-1.5B (r32), c1r8-1.5B (r8).
Fits saturating acc(r)=A - B*r^-c per arm; reports asymptote A, the clean-noisy gap,
and whether compute closes the gap. Uses whatever seeds have finished.
Usage: python jobscripts/fit_c2.py
"""
import os, re, statistics as st
from collections import defaultdict
import wandb
import numpy as np

ENTITY = os.environ.get("WANDB_ENTITY", "rm4411-princeton-university")
api = wandb.Api()
runs = api.runs(f"{ENTITY}/RLVR")

NAME_R = re.compile(r"-r(\d+)-s\d+$")

# Dedup: killed-then-resubmitted runs share a name; keep the one with max _step
# (the completed one). Truncated duplicates otherwise poison cell means/variance.
best_by_name = {}
for r in runs:
    n = r.name
    is_c2 = n.startswith("c2-gsm8k-1.5B")
    is_c1 = n.startswith("c1-1.5B") or n.startswith("c1r8-1.5B")
    if not (is_c2 or is_c1):
        continue
    em = r.config.get("env", {}).get("math", {})
    fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    if fp is None or acc is None:
        continue
    fp, fn = round(float(fp), 2), round(float(fn), 2)
    if (fp, fn) not in [(0.0, 0.0), (0.2, 0.2)]:
        continue
    mm = NAME_R.search(n)
    if not mm:
        continue
    step = r.summary.get("_step", 0) or 0
    prev = best_by_name.get(n)
    if prev is None or step > prev[0]:
        best_by_name[n] = (step, fp, fn, int(mm.group(1)), float(acc))

# arm key (fp,fn) -> r -> list of final acc (one per seed, completed run only)
arms = defaultdict(lambda: defaultdict(list))
for step, fp, fn, rr, acc in best_by_name.values():
    arms[(fp, fn)][rr].append(acc)


def rmean(vals):
    """mean excluding collapsed (~0) seeds."""
    ok = [v for v in vals if v >= 0.10]
    return st.mean(ok if ok else vals)


def sat(r, A, B, c):
    return A - B * np.power(r, -c)


def fit_arm(rs, ys):
    """3-param saturating fit; bootstrap asymptote CI over the r-point means."""
    from scipy.optimize import curve_fit
    rs = np.array(rs, float); ys = np.array(ys, float)
    p0 = [max(ys), max(0.05, max(ys) - min(ys)), 0.5]
    bounds = ([0.0, 0.0, 0.05], [1.0, 2.0, 3.0])
    popt, _ = curve_fit(sat, rs, ys, p0=p0, bounds=bounds, maxfev=20000)
    # in-sample R2
    pred = sat(rs, *popt)
    ss_res = np.sum((ys - pred) ** 2); ss_tot = np.sum((ys - ys.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    return popt, r2


print(f"{'='*70}\nC2 COMPUTE AXIS — acc vs rollouts r (GSM8K 1.5B)\n{'='*70}")
summary = {}
for arm in [(0.0, 0.0), (0.2, 0.2)]:
    d = arms.get(arm, {})
    if not d:
        print(f"\narm {arm}: NO DATA yet"); continue
    rs = sorted(d)
    print(f"\narm (fp={arm[0]}, fn={arm[1]}):  {sum(len(v) for v in d.values())} runs, r-points {rs}")
    xs, ys = [], []
    for rr in rs:
        vals = d[rr]
        # a fully-trained run at ~0 = genuine training collapse (not truncation, already deduped);
        # exclude from the central tendency but report the collapse count — it's itself a finding.
        ok = [v for v in vals if v >= 0.10]
        ncol = len(vals) - len(ok)
        use = ok if ok else vals
        mean = st.mean(use); sd = st.pstdev(use) if len(use) > 1 else 0.0
        col = f"  [{ncol} COLLAPSED→~0 excluded]" if ncol else ""
        print(f"  r={rr:>3}: acc={mean:.3f}±{sd:.3f} (n={len(use)}){col}")
        xs.append(rr); ys.append(mean)
    if len(xs) >= 4:
        try:
            (A, B, c), r2 = fit_arm(xs, ys)
            print(f"  FIT acc(r)={A:.3f} - {B:.3f}*r^-{c:.2f}   (asymptote A={A:.3f}, R2_in={r2:.3f})")
            summary[arm] = dict(A=A, B=B, c=c, xs=xs, ys=ys)
        except Exception as e:
            print(f"  fit failed ({e}); need more r-points")
    else:
        print(f"  only {len(xs)} r-points — need >=4 for the 3-param saturating fit")
        summary[arm] = dict(xs=xs, ys=ys)

# clean vs noisy: does compute close the gap?
if (0.0, 0.0) in summary and (0.2, 0.2) in summary:
    cA = summary[(0.0, 0.0)].get("A"); nA = summary[(0.2, 0.2)].get("A")
    print(f"\n{'-'*70}\nCLEAN vs NOISY (0.2,0.2):")
    # gap at each shared r
    cd = arms[(0.0, 0.0)]; nd = arms[(0.2, 0.2)]
    shared = sorted(set(cd) & set(nd))
    if shared:
        print("  finite-r gap (clean - noisy, collapse-robust):")
        for rr in shared:
            g = rmean(cd[rr]) - rmean(nd[rr])
            print(f"    r={rr:>3}: clean={rmean(cd[rr]):.3f}  noisy={rmean(nd[rr]):.3f}  gap={g:+.3f}")
    if cA is not None and nA is not None:
        print(f"  ASYMPTOTE gap: clean A={cA:.3f}  noisy A={nA:.3f}  gap={cA-nA:+.3f}")
        # margin m=1-.2-.2=.6 -> if noise were pure margin-scaling, expect noisy ceiling ~ m*clean?
        print(f"  => does compute close the noise gap? {'shrinks with r' if shared and (st.mean(cd[shared[0]])-st.mean(nd[shared[0]])) > (cA-nA) else 'gap persists at asymptote'}")
