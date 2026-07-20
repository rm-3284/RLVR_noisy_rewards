"""Cross-family margin-collapse test: OLMo-2-1B on GSM8K, full 4x4 (fp,fn) grid at r=8.

This is the SECOND family (Qwen is the first) and — unlike OLMo-1B/MATH, which is floored at
~2-8% and so cannot resolve a dose-response — GSM8K is unfloored (base ~0.44), so the noise
penalty is measurable across the whole grid.

The grid is 16 cells: fp,fn in {0, 0.15, 0.3, 0.45}, 5 seeds each. 12 cells are OFF-DIAGONAL
(fp != fn), which is what makes the collapse falsifiable at all: on the diagonal fp-fn == 0
everywhere, so the asymmetry term is unidentifiable.

  m = 1 - fp - fn        (margin; the claimed sufficient statistic)
  d = fp - fn            (asymmetry; must be IRRELEVANT if strict collapse holds)

  M1 (strict collapse):  logit(acc) = a + b*m
  M2 (asymmetry):        logit(acc) = a + b*m + g*d

PRE-COMMITTED PREDICTION (from the base-rate finding, see memory/key-finding-base-rate.md):
the FP/FN asymmetry is base-rate-moderated, NOT task-moderated. GSM8K/OLMo-1B has a HIGH base
rate (~0.44), so the prediction is g ~= 0 (collapse holds here), whereas the low-base configs
(e.g. MATH-1.5B) showed g > 0. A large g here would FALSIFY the base-rate account.

Reported: per-cell means, the direct matched-margin test (do cells at equal m coincide?),
g with a seed-level bootstrap CI, and leave-one-CELL-out held-out RMSE for M1 vs M2.
"""
import re, sys
import numpy as np
from collections import defaultdict
import wandb

RGX = r"^aiOLMOgsm-1B-r8-fp"
PROJECT = "rm4411-princeton-university/RLVR"
RNG = np.random.default_rng(0)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- pull runs
api = wandb.Api()
best = {}  # run name -> (step, fp, fn, acc)
for r in api.runs(PROJECT, filters={"display_name": {"$regex": RGX}}, per_page=500):
    em = r.config.get("env", {}).get("math", {})
    fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    if fp is None or fn is None or acc is None:
        continue
    step = r.summary.get("_step", 0) or 0
    # dedup killed+resubmitted duplicates by keeping the furthest-trained run of each name
    if r.name not in best or step > best[r.name][0]:
        best[r.name] = (step, round(float(fp), 2), round(float(fn), 2), float(acc))

if not best:
    sys.exit(f"NO RUNS matched {RGX} in {PROJECT} — are they synced to the cloud?")

maxstep = max(v[0] for v in best.values())
seeds = defaultdict(list)  # (fp,fn) -> [acc per seed]
dropped = 0
for name, (step, fp, fn, acc) in best.items():
    if step < 0.6 * maxstep:  # drop undertrained/crashed runs
        dropped += 1
        continue
    seeds[(fp, fn)].append(acc)

print(f"pulled {len(best)} runs (maxstep={maxstep}); dropped {dropped} undertrained; "
      f"{len(seeds)} cells")
for k in sorted(seeds):
    if len(seeds[k]) != 5:
        print(f"  NOTE cell fp={k[0]} fn={k[1]} has {len(seeds[k])} seeds (expected 5)")

# ---------------------------------------------------------------- cell table
print("\n=== cell means (r=8, OLMo-2-1B / GSM8K) ===")
print(f"{'fp':>5} {'fn':>5} {'m':>6} {'d':>6} {'acc':>7} {'sd':>6} {'n':>3}")
cells = {}
for (fp, fn) in sorted(seeds, key=lambda t: (-(1 - t[0] - t[1]), t[0])):
    a = np.array(seeds[(fp, fn)])
    m, d = 1 - fp - fn, fp - fn
    cells[(fp, fn)] = (m, d, a.mean(), a.std(ddof=1) if len(a) > 1 else np.nan, len(a))
    print(f"{fp:>5} {fn:>5} {m:>6.2f} {d:>6.2f} {a.mean():>7.4f} "
          f"{(a.std(ddof=1) if len(a)>1 else float('nan')):>6.4f} {len(a):>3}")

# typical within-cell seed noise — the yardstick for "do matched cells coincide?"
seed_sd = np.nanmean([v[3] for v in cells.values()])
print(f"\nmean within-cell seed SD = {seed_sd:.4f}  (this is the noise floor)")

# ------------------------------------------- direct matched-margin test
print("\n=== MATCHED-MARGIN TEST (the falsification test) ===")
print("If accuracy collapses onto m, cells sharing a margin must coincide to within seed noise.")
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
    cellstr = ", ".join(f"({fp},{fn})={mu:.3f}" for fp, fn, mu in sorted(grp))
    flag = "  <-- EXCEEDS seed noise" if spread > 2 * seed_sd else ""
    print(f"  m={m:<5} spread={spread:.4f}  {cellstr}{flag}")
print(f"\nworst matched-margin spread = {worst:.4f} vs 2x seed noise = {2*seed_sd:.4f}")
print("VERDICT:", "consistent with collapse" if worst <= 2 * seed_sd
      else "COLLAPSE VIOLATED at this margin resolution")

# ---------------------------------------------------------------- M1 vs M2
X_keys = sorted(cells)
m_ = np.array([cells[k][0] for k in X_keys])
d_ = np.array([cells[k][1] for k in X_keys])
y_ = logit([cells[k][2] for k in X_keys])

def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

X1 = np.c_[np.ones_like(m_), m_]
X2 = np.c_[np.ones_like(m_), m_, d_]
b1, b2 = fit(X1, y_), fit(X2, y_)
print("\n=== M1 (collapse) vs M2 (+asymmetry) — in-sample ===")
print(f"  M1: logit(acc) = {b1[0]:+.3f} {b1[1]:+.3f}*m")
print(f"  M2: logit(acc) = {b2[0]:+.3f} {b2[1]:+.3f}*m {b2[2]:+.3f}*d   <-- g = {b2[2]:+.4f}")

# CI on g.
#
# DO NOT resample only seeds-within-cells and refit on cell means: that propagates ONLY seed
# noise (~0.12 logit here) and ignores the systematic CELL-TO-CELL scatter (~0.28 logit) that
# actually dominates the residual. It understated SE(g) by ~3x and produced a bogus
# "collapse FALSIFIED" verdict on 2026-07-14. The unit of resampling must be the CELL.
#
# Two estimators, both cell-level:
#   (a) seed-level OLS (80 obs) -> analytic SE, correct error structure
#   (b) cluster bootstrap over CELLS -> nonparametric, robust to lack-of-fit
nseed = np.array([len(seeds[k]) for k in X_keys])
M_s = np.repeat(m_, nseed)
D_s = np.repeat(d_, nseed)
Y_s = logit(np.concatenate([seeds[k] for k in X_keys]))
X2s = np.c_[np.ones_like(M_s), M_s, D_s]
b2s = fit(X2s, Y_s)
resid_sd = np.std(Y_s - X2s @ b2s, ddof=3)
se_g = resid_sd * np.sqrt(np.linalg.inv(X2s.T @ X2s)[2, 2])
lo_a, hi_a = b2s[2] - 1.96 * se_g, b2s[2] + 1.96 * se_g
print(f"  (a) seed-level OLS: g = {b2s[2]:+.4f}  SE = {se_g:.4f}  "
      f"95% CI [{lo_a:+.4f}, {hi_a:+.4f}]")

gs = []
idx = np.arange(len(X_keys))
for _ in range(2000):
    take = RNG.choice(idx, size=len(idx), replace=True)
    gs.append(fit(X2[take], y_[take])[2])
lo_b, hi_b = np.percentile(gs, [2.5, 97.5])
print(f"  (b) cluster bootstrap over CELLS: 95% CI [{lo_b:+.4f}, {hi_b:+.4f}]")

if lo_a <= 0 <= hi_a:
    print(f"  -> g NOT distinguishable from 0. Strict collapse survives; any asymmetry is\n"
          f"     bounded by |g| < {max(abs(lo_a), abs(hi_a)):.2f}, i.e. < "
          f"{100*max(abs(lo_a),abs(hi_a))/abs(b2s[1]):.0f}% of the margin slope (b={b2s[1]:+.2f}).")
    print(f"     NOTE: this is a BOUND, not a confirmation of g==0 — see "
          f"analysis/power_check_asymmetry.py (we can only resolve |g| >~ 0.5).")
else:
    print("  -> g significantly nonzero at the CELL level: strict collapse genuinely violated.")

# ------------------------------------- leave-one-CELL-out held-out RMSE
print("\n=== held-out (leave-one-CELL-out) RMSE ===")
errs = {1: [], 2: []}
for i in range(len(X_keys)):
    tr = np.ones(len(X_keys), bool); tr[i] = False
    for tag, X in ((1, X1), (2, X2)):
        beta = fit(X[tr], y_[tr])
        errs[tag].append(y_[i] - X[i] @ beta)
r1 = np.sqrt(np.mean(np.square(errs[1])))
r2 = np.sqrt(np.mean(np.square(errs[2])))
print(f"  M1 (collapse):   LOCO RMSE = {r1:.4f} (logit units)")
print(f"  M2 (+asymmetry): LOCO RMSE = {r2:.4f} (logit units)")
print("  ->", "asymmetry term does NOT earn its keep out-of-sample (collapse preferred)"
      if r2 >= r1 else "asymmetry term IMPROVES held-out prediction (collapse insufficient)")

base = cells.get((0.0, 0.0), (None, None, float('nan')))[2]
print(f"\nbase rate (clean, fp=fn=0) = {base:.4f}  "
      f"[high-base regime -> pre-committed prediction was g ~= 0]")
