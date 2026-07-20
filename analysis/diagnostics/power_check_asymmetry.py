"""Is the 16-cell grid POWERED to detect an FP/FN asymmetry, or is 'collapse preferred' just
a small-n artifact?

Context: on OLMo-1B/GSM8K r=8 (16 cells x 5 seeds) we get a CONFLICT --
  * bootstrap CI on g excludes 0  (g = -0.136, [-0.228, -0.042])
  * but leave-one-CELL-out RMSE PREFERS the no-asymmetry model (M1 0.270 < M2 0.282)
Two readings: (a) g is real but tiny, and LOCO with 16 points can't pay for a 3rd parameter;
(b) g is noise and the bootstrap CI is overconfident. This script distinguishes them.

Method: take the ACTUAL cell layout and ACTUAL seed noise. Plant a known g_true, simulate
seed-level accuracies, and ask how often each test recovers it. That calibrates BOTH tests
instead of trusting either.

  M1: logit(acc) = a + b*m
  M2: logit(acc) = a + b*m + g*d      m = 1-fp-fn,  d = fp-fn

Also refits on SEED-LEVEL data (80 obs, not 16 cell means), which is the statistically correct
error structure and has more power than the cell-mean regression.
"""
import numpy as np
from collections import defaultdict
import wandb

RGX = r"^aiOLMOgsm-1B-r8-fp"
PROJECT = "rm4411-princeton-university/RLVR"
RNG = np.random.default_rng(0)
NSIM = 600


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def invlogit(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ------------------------------------------------------------------ real data
api = wandb.Api()
best = {}
for r in api.runs(PROJECT, filters={"display_name": {"$regex": RGX}}, per_page=500):
    em = r.config.get("env", {}).get("math", {})
    fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    if fp is None or fn is None or acc is None:
        continue
    step = r.summary.get("_step", 0) or 0
    if r.name not in best or step > best[r.name][0]:
        best[r.name] = (step, round(float(fp), 2), round(float(fn), 2), float(acc))

seeds = defaultdict(list)
for _, (step, fp, fn, acc) in best.items():
    seeds[(fp, fn)].append(acc)
keys = sorted(seeds)
print(f"cells={len(keys)}  runs={sum(len(v) for v in seeds.values())}")

m_ = np.array([1 - fp - fn for fp, fn in keys])
d_ = np.array([fp - fn for fp, fn in keys])
nseed = np.array([len(seeds[k]) for k in keys])

# seed-level design (80 rows)
M_s = np.repeat(m_, nseed)
D_s = np.repeat(d_, nseed)
Y_s = logit(np.concatenate([seeds[k] for k in keys]))
CELL_s = np.repeat(np.arange(len(keys)), nseed)

X1s = np.c_[np.ones_like(M_s), M_s]
X2s = np.c_[np.ones_like(M_s), M_s, D_s]

b1s, b2s = fit(X1s, Y_s), fit(X2s, Y_s)
resid_sd = np.std(Y_s - X2s @ b2s, ddof=3)   # seed-level noise in LOGIT units
print(f"\n=== seed-level fit (80 obs, correct error structure) ===")
print(f"  M1: a={b1s[0]:+.3f} b={b1s[1]:+.3f}")
print(f"  M2: a={b2s[0]:+.3f} b={b2s[1]:+.3f}  g={b2s[2]:+.4f}")
print(f"  seed-level residual SD (logit) = {resid_sd:.4f}")

# analytic SE of g at seed level
XtX_inv = np.linalg.inv(X2s.T @ X2s)
se_g = resid_sd * np.sqrt(XtX_inv[2, 2])
print(f"  SE(g) = {se_g:.4f}   ->  g/SE = {b2s[2]/se_g:+.2f}   "
      f"95% CI [{b2s[2]-1.96*se_g:+.4f}, {b2s[2]+1.96*se_g:+.4f}]")

# ---------------------------------------- LOCO helper (fit on seeds, predict held-out cell)
def loco_rmse(Y, X1, X2, cell_ids, ncell):
    e1, e2 = [], []
    for c in range(ncell):
        tr, te = cell_ids != c, cell_ids == c
        for X, e in ((X1, e1), (X2, e2)):
            beta = fit(X[tr], Y[tr])
            e.append(Y[te].mean() - X[te][0] @ beta)   # predict the held-out cell's mean
    return np.sqrt(np.mean(np.square(e1))), np.sqrt(np.mean(np.square(e2)))


r1, r2 = loco_rmse(Y_s, X1s, X2s, CELL_s, len(keys))
print(f"\n=== LOCO on REAL data (seed-level fit, cell-level holdout) ===")
print(f"  M1 = {r1:.4f}   M2 = {r2:.4f}   -> {'M1 (collapse)' if r1 <= r2 else 'M2 (asymmetry)'} preferred")

# ------------------------------------------------------------- POWER SIMULATION
print(f"\n=== POWER: plant a known g_true, can we recover it? ({NSIM} sims each) ===")
print("  g_true |  LOCO picks M2 |  CI excludes 0 |  mean g_hat")
print("  " + "-" * 60)
truth = np.array([0.0, -0.05, -0.136, -0.25, -0.5, -1.0])
a0, b0 = b2s[0], b2s[1]
for g_true in truth:
    loco_hits = ci_hits = 0
    ghats = []
    for _ in range(NSIM):
        mu = a0 + b0 * M_s + g_true * D_s
        Y = mu + RNG.normal(0, resid_sd, size=M_s.shape)
        bh = fit(X2s, Y)
        ghats.append(bh[2])
        se = np.std(Y - X2s @ bh, ddof=3) * np.sqrt(XtX_inv[2, 2])
        if abs(bh[2]) > 1.96 * se:
            ci_hits += 1
        q1, q2 = loco_rmse(Y, X1s, X2s, CELL_s, len(keys))
        if q2 < q1:
            loco_hits += 1
    print(f"  {g_true:+6.3f} |     {100*loco_hits/NSIM:5.1f}%     |    {100*ci_hits/NSIM:5.1f}%     |  {np.mean(ghats):+.4f}")

print("""
READ THIS TABLE AS:
 * 'LOCO picks M2' at g_true=0 is the FALSE-POSITIVE rate; at g_true<0 it is POWER.
 * If LOCO's power is low even at LARGE g_true, then 'collapse preferred' is WEAK evidence
   (the test simply cannot see an asymmetry with 16 cells) -- we do NOT have enough cells.
 * If LOCO's power is HIGH at moderate g_true but our observed g=-0.136 sits below that,
   then we HAVE enough cells and the honest conclusion is that any asymmetry is genuinely
   SMALL -- not that we failed to look.""")
