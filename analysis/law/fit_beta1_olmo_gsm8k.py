"""Closed-form tradeoff law on the COMPLETE OLMo-1B/GSM8K cube (second family).

Fits:   logit(acc) = a0 + a1*log2(r) - beta*(1-m) - beta1*(1-m)*log2(r)
  m = 1 - fp - fn (margin), r in {8,32,128} (rollouts).
  a0    = clean intercept
  a1    = compute slope (rollouts help clean)
  beta  = noise penalty (margin loss hurts)
  beta1 = COMPUTE x NOISE interaction — does more compute buy back noise robustness?

beta1 is the headline: beta1 > 0 means extra rollouts partially offset reward noise (a real
compute-vs-verifier-quality tradeoff to prescribe); beta1 ~ 0 means noise is a fixed tax
compute can't fix. This is the second-family estimate with all 3 rollout points (vs the
earlier 2-point preliminary).

Reports coefficients, seed-level + cluster-bootstrap CI on beta1, and leave-one-CELL-out
held-out RMSE of the full model vs a no-interaction (beta1=0) restriction.
"""
import re
from collections import defaultdict
import numpy as np
import wandb

PROJECT = "rm4411-princeton-university/RLVR"
RNG = np.random.default_rng(0)
api = wandb.Api()


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


# pull the whole cube, key by (fp,fn,r) -> [acc per seed] at converged step (per rollout)
best = {}
for r in api.runs(PROJECT, filters={"display_name": {"$regex": "^aiOLMOgsm-1B-r"}}, per_page=500):
    m = re.search(r"-r(\d+)-fp", r.name)
    if not m:
        continue
    roll = int(m.group(1))
    em = r.config.get("env", {}).get("math", {})
    fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    if fp is None or fn is None or acc is None:
        continue
    step = r.summary.get("_step", 0) or 0
    key = r.name
    if key not in best or step > best[key][0]:
        best[key] = (step, roll, round(float(fp), 2), round(float(fn), 2), float(acc))

# converged filter PER rollout (each r has its own max step)
maxstep = defaultdict(int)
for _, (step, roll, fp, fn, acc) in best.items():
    maxstep[roll] = max(maxstep[roll], step)
seeds = defaultdict(list)   # (fp,fn,roll) -> [acc]
for _, (step, roll, fp, fn, acc) in best.items():
    if step < 0.6 * maxstep[roll]:
        continue
    seeds[(fp, fn, roll)].append(acc)

cells = sorted(seeds)
print(f"pulled cube: {sum(len(v) for v in seeds.values())} runs, {len(cells)} (fp,fn,r) cells")
byroll = defaultdict(int)
for (fp, fn, roll) in cells:
    byroll[roll] += 1
print("  cells per rollout:", dict(sorted(byroll.items())))

# design matrix at cell level (seed-averaged), and seed level for CI
def design(fp, fn, roll):
    m = 1 - fp - fn
    lr = np.log2(roll)
    return [1.0, lr, -(1 - m), -(1 - m) * lr]   # [a0, a1, beta, beta1]

Xc = np.array([design(*k) for k in cells])
yc = logit([np.mean(seeds[k]) for k in cells])
b = fit(Xc, yc)
print("\n=== closed-form fit (cell-level) ===")
print(f"  a0 (clean intercept) = {b[0]:+.3f}")
print(f"  a1 (compute slope)   = {b[1]:+.3f}")
print(f"  beta (noise penalty) = {b[2]:+.3f}")
print(f"  beta1 (compute x noise) = {b[3]:+.4f}   <-- HEADLINE")

# seed-level fit + analytic CI on beta1
rows, Y = [], []
for k in cells:
    for a in seeds[k]:
        rows.append(design(*k)); Y.append(a)
Xs = np.array(rows); Ys = logit(Y)
bs = fit(Xs, Ys)
resid_sd = np.std(Ys - Xs @ bs, ddof=4)
se_b1 = resid_sd * np.sqrt(np.linalg.inv(Xs.T @ Xs)[3, 3])
lo_a, hi_a = bs[3] - 1.96 * se_b1, bs[3] + 1.96 * se_b1
print(f"\n  seed-level beta1 = {bs[3]:+.4f}  SE {se_b1:.4f}  95% CI [{lo_a:+.4f}, {hi_a:+.4f}]")

# cluster bootstrap over CELLS
b1s = []
idx = np.arange(len(cells))
for _ in range(2000):
    take = RNG.choice(idx, size=len(idx), replace=True)
    Xb = Xc[take]; yb = yc[take]
    b1s.append(fit(Xb, yb)[3])
lo_b, hi_b = np.percentile(b1s, [2.5, 97.5])
print(f"  cluster-bootstrap beta1 95% CI [{lo_b:+.4f}, {hi_b:+.4f}]")
sign = "beta1 > 0: MORE COMPUTE BUYS NOISE ROBUSTNESS" if lo_a > 0 else \
       ("beta1 < 0: compute AMPLIFIES noise harm" if hi_a < 0 else
        "beta1 NOT distinguishable from 0: noise & compute ~separable")
print(f"  -> {sign}")

# leave-one-CELL-out: full model vs beta1=0 restriction
print("\n=== held-out (leave-one-CELL-out) RMSE ===")
X_full, X_rest = Xc, Xc[:, :3]   # restricted drops the interaction column
ef, er = [], []
for i in range(len(cells)):
    tr = np.ones(len(cells), bool); tr[i] = False
    ef.append(yc[i] - X_full[i] @ fit(X_full[tr], yc[tr]))
    er.append(yc[i] - X_rest[i] @ fit(X_rest[tr], yc[tr]))
rf = np.sqrt(np.mean(np.square(ef))); rr = np.sqrt(np.mean(np.square(er)))
print(f"  full (with beta1):  RMSE = {rf:.4f}")
print(f"  restricted (beta1=0): RMSE = {rr:.4f}")
print("  ->", "interaction EARNS its keep (compute-noise coupling is real)"
      if rf < rr else "interaction does NOT improve held-out fit (noise ~ additive tax)")
