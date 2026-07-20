"""Cross-family margin-collapse test on the COMPLETE OLMo-1B/GSM8K cube, at EACH rollout count.

Earlier (fit_collapse_olmo_gsm8k.py) only tested r=8. Now the full cube is done (240/240,
80 runs per rollout), so we test whether the collapse holds at r=8, r=32, AND r=128 — i.e.
whether "accuracy depends only on margin m = 1-fp-fn" survives across compute scales on the
second family.

For each rollout count independently: pull the 16-cell grid, run the matched-margin test
(cells sharing m must coincide within seed noise), and report the seed-level asymmetry CI.
"""
import re, sys, statistics as st
from collections import defaultdict
import numpy as np
import wandb

PROJECT = "rm4411-princeton-university/RLVR"
api = wandb.Api()


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def pull(rollout_tag):
    """rollout_tag like 'r8'. Returns {(fp,fn): [acc per seed]} at converged step."""
    best = {}
    rgx = rf"^aiOLMOgsm-1B-{rollout_tag}-fp"
    for r in api.runs(PROJECT, filters={"display_name": {"$regex": rgx}}, per_page=500):
        em = r.config.get("env", {}).get("math", {})
        fp, fn = em.get("fp"), em.get("fn")
        acc = r.summary.get("validation/accuracy")
        if fp is None or fn is None or acc is None:
            continue
        step = r.summary.get("_step", 0) or 0
        if r.name not in best or step > best[r.name][0]:
            best[r.name] = (step, round(float(fp), 2), round(float(fn), 2), float(acc))
    if not best:
        return {}
    maxstep = max(v[0] for v in best.values())
    seeds = defaultdict(list)
    for step, fp, fn, acc in best.values():
        if step < 0.6 * maxstep:   # drop undertrained/crashed
            continue
        seeds[(fp, fn)].append(acc)
    return seeds


for tag in ["r8", "r32", "r128"]:
    seeds = pull(tag)
    print(f"\n{'='*70}\n=== OLMo-1B / GSM8K  {tag}  ({sum(len(v) for v in seeds.values())} runs, "
          f"{len(seeds)} cells) ===")
    if len(seeds) < 8:
        print(f"  only {len(seeds)} cells — skipping (incomplete)")
        continue

    # cell table
    cells = {}
    for (fp, fn), a in seeds.items():
        a = np.array(a)
        cells[(fp, fn)] = (1 - fp - fn, fp - fn, a.mean(), a.std(ddof=1) if len(a) > 1 else np.nan, len(a))
    seed_sd = np.nanmean([v[3] for v in cells.values()])
    print(f"  mean within-cell seed SD = {seed_sd:.4f}  (noise floor)")

    # matched-margin test
    bym = defaultdict(list)
    for (fp, fn), (m, d, mu, sd, n) in cells.items():
        bym[round(m, 2)].append((fp, fn, mu))
    worst = 0.0
    for m in sorted(bym, reverse=True):
        grp = bym[m]
        if len(grp) < 2:
            continue
        mus = np.array([g[2] for g in grp])
        spread = mus.max() - mus.min()
        worst = max(worst, spread)
        flag = "  <-- EXCEEDS 2x seed noise" if spread > 2 * seed_sd else ""
        print(f"    m={m:<5} spread={spread:.4f}  " +
              ", ".join(f"({fp},{fn})={mu:.3f}" for fp, fn, mu in sorted(grp)) + flag)
    print(f"  worst matched-margin spread = {worst:.4f} vs 2x seed noise = {2*seed_sd:.4f}")
    print("  VERDICT:", "CONSISTENT with collapse" if worst <= 2 * seed_sd
          else "collapse VIOLATED at this resolution")

    # seed-level asymmetry CI (correct error structure)
    keys = sorted(cells)
    nseed = np.array([len(seeds[k]) for k in keys])
    M = np.repeat([cells[k][0] for k in keys], nseed)
    D = np.repeat([cells[k][1] for k in keys], nseed)
    Y = logit(np.concatenate([seeds[k] for k in keys]))
    X2 = np.c_[np.ones_like(M), M, D]
    b2 = fit(X2, Y)
    resid_sd = np.std(Y - X2 @ b2, ddof=3)
    se_g = resid_sd * np.sqrt(np.linalg.inv(X2.T @ X2)[2, 2])
    lo, hi = b2[2] - 1.96 * se_g, b2[2] + 1.96 * se_g
    print(f"  margin slope b={b2[1]:+.3f} ; asymmetry g={b2[2]:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")
    print("  ->", "g indistinguishable from 0: strict collapse survives"
          if lo <= 0 <= hi else "g nonzero at cell level")

    base = cells.get((0.0, 0.0), (0, 0, float('nan')))[2]
    print(f"  base rate (fp=fn=0) = {base:.4f}")
