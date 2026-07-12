"""EXPLICIT CLOSED-FORM law fit. NO free per-r intercepts (that was the cheat).

  logit(acc) = a0 + a1*log2(r) - beta*(1-m) - beta1*(1-m)*log2(r)      m = 1 - fp - fn

Four coefficients, every term parametric. Plug in (r,m) -> acc. Like Chinchilla L(N,D).
Held-out tests:
  - LOCO   : leave-one-CELL-out (drop one (r,m), refit 4 coefs, predict it)
  - LORO   : leave-one-R-out    (drop ALL cells at one r, refit, predict them = extrapolate
             the COMPUTE axis to an unseen rollout count. The free-intercept fit could not
             do this at all -- no intercept for an unseen r.)
  - vs separable null (beta1=0, 3 coefs) on the same held-out folds.
Reads offline .wandb directly; dedup by seed (fullest run per seed); >=10-eval gate. m=1-fp-fn (fp,fn independent).
"""
import glob, os, math, re
import numpy as np
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

def keyof(it): return it.key or ("/".join(it.nested_key) if it.nested_key else "")
def pick_run(d):
    # DEDUP: if a seed-dir has multiple exp runs (e.g. v4o1 clean-anchor has 2), pick the one
    # with the MOST validation evals (fullest), NOT newest mtime — avoids reading a truncated dup.
    r = glob.glob(f"{d}/exp_*/wandb/wandb/offline-run-*/run-*.wandb")
    if not r: return None
    return r[0] if len(r) == 1 else max(r, key=lambda f: len(series(f)))
def series(f):
    ds = DataStore(); ds.open_for_scan(f); out = []
    while True:
        try: rb = ds.scan_data()
        except: break
        if rb is None: break
        r = pb.Record()
        try: r.ParseFromString(rb)
        except: continue
        if r.WhichOneof("record_type") == "history":
            d = {keyof(i): i.value_json for i in r.history.item}
            if "validation/accuracy" in d:
                try: out.append((int(d.get("_step", "-1")), float(d["validation/accuracy"])))
                except: pass
    out.sort(); return out

# Patterns now use independent {fp} and {fn} (was fn=={fp}). Diagonal models just pass fp==fn.
MODELS = {
 "MATH-0.5B": ("logs/ts05M-0.5B-{r}-fp{fp}-fn{fn}-s*", "logs/aiMATH05hi-0.5B-{r}-fp{fp}-fn{fn}-s*"),
 "MATH-1.5B": ("logs/tsM-1.5B-{r}-fp{fp}-fn{fn}-s*", "logs/tsMb-1.5B-{r}-fp{fp}-fn{fn}-s*"),
 "GSM8K-0.5B": ("logs/ct05-gsm8k-0.5B-fp{fp}-fn{fn}-{r}-s*", "logs/p05B-gsm8k-fp{fp}-fn{fn}-{r}-s*"),
 "GSM8K-1.5B": ("logs/c2-gsm8k-1.5B-fp{fp}-fn{fn}-{r}-s*", "logs/c2b-gsm8k-1.5B-fp{fp}-fn{fn}-{r}-s*",
                "logs/c2c-gsm8k-1.5B-fp{fp}-fn{fn}-{r}-s*", "logs/c1r8-1.5B-fp{fp}-fn{fn}-{r}-s*",
                "logs/c1-1.5B-fp{fp}-fn{fn}-{r}-s*"),
 "MATH-OLMo1B": ("logs/v4o1-{r}-fp{fp}-fn{fn}-s*",),   # CROSS-FAMILY: OLMo-2-1B, full off-diagonal grid
}
RS  = {"MATH-0.5B": [8, 16, 32, 64], "MATH-1.5B": [8, 16, 32, 64, 128],
       "GSM8K-0.5B": [8, 32, 128], "GSM8K-1.5B": [4, 8, 16, 32, 64, 128, 256],
       "MATH-OLMo1B": [8, 32, 128]}
# per-model list of (fp, fn) noise cells. Diagonal models = (x,x); OLMo-1B = full fp x fn grid
# (the >=4-seed gate below silently drops combos that haven't converged yet).
_diag = lambda xs: [(x, x) for x in xs]
NOISE = {"MATH-0.5B": _diag([0.0, 0.15, 0.3]), "MATH-1.5B": _diag([0.0, 0.15, 0.3]),
         "GSM8K-0.5B": _diag([0.0, 0.15, 0.2, 0.3]), "GSM8K-1.5B": _diag([0.0, 0.2, 0.3]),
         "MATH-OLMo1B": [(fp, fn) for fp in [0.0, 0.15, 0.3, 0.45] for fn in [0.0, 0.15, 0.3, 0.45]]}

def cellmean(model, r, fp, fn):
    per_seed = {}
    for pat in MODELS[model]:
        for d in glob.glob(pat.format(r=f"r{r}", fp=fp, fn=fn)):
            f = pick_run(d)
            if not f: continue
            s = series(f)
            if len(s) >= 10:
                sm = re.search(r'-s(\d+)$', d); sid = sm.group(1) if sm else d
                per_seed.setdefault(sid, []).append(float(np.mean([a for _, a in s[-5:]])))
    return [float(np.mean(v)) for v in per_seed.values()]

def logit(a): a = min(max(a, 1e-4), 1 - 1e-4); return math.log(a / (1 - a))

def row_of(r, m, interaction):
    L = math.log2(r)
    x = [1.0, L, -(1 - m)]
    if interaction: x.append(-(1 - m) * L)
    return x

def lstsq(X, y): c, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None); return c
def predict(c, r, m, interaction):
    lg = np.array(row_of(r, m, interaction)) @ c
    return 1 / (1 + math.exp(-lg))

for model in MODELS:
    print(f"\n=== {model} ===  (CLOSED FORM: 4 coefficients, no free intercepts)")
    # build (r, m, acc) cells
    cells = []   # (r, m, acc)
    for r in RS[model]:
        for (fp, fn) in NOISE[model]:
            sd = cellmean(model, r, fp, fn)
            if len(sd) >= 4:
                cells.append((r, 1 - fp - fn, float(np.mean(sd))))
    rs = sorted({r for r, _, _ in cells}); ms = sorted({m for _, m, _ in cells})
    print(f"  grid: {len(cells)} cells, r={rs}, m={[round(x,2) for x in ms]}")

    # ---- full-grid closed-form fit (interaction) ----
    Xi = [row_of(r, m, True) for r, m, _ in cells]; yi = [logit(a) for *_, a in cells]
    ci = lstsq(Xi, yi)
    a0, a1, beta, beta1 = ci
    in_mae = np.mean([abs(predict(ci, r, m, True) - a) for r, m, a in cells])
    print(f"  FULL FIT:  a0={a0:+.3f}  a1={a1:+.3f}  beta={beta:+.3f}  beta1={beta1:+.3f}   in-MAE={in_mae:.4f}")

    # ---- LOCO: leave-one-cell-out, interaction vs separable ----
    def loco(interaction):
        errs = []
        for i in range(len(cells)):
            tr = [cells[j] for j in range(len(cells)) if j != i]
            c = lstsq([row_of(r, m, interaction) for r, m, _ in tr], [logit(a) for *_, a in tr])
            r, m, a = cells[i]; errs.append(abs(predict(c, r, m, interaction) - a))
        return np.mean(errs)
    li, ls = loco(True), loco(False)

    # ---- LORO: leave-one-R-out (extrapolate compute), interaction vs separable ----
    def loro(interaction):
        errs = []
        for rh in rs:
            tr = [c for c in cells if c[0] != rh]; te = [c for c in cells if c[0] == rh]
            if len({c[0] for c in tr}) < (3 if interaction else 2): continue  # need enough r-span for a1
            c = lstsq([row_of(r, m, interaction) for r, m, _ in tr], [logit(a) for *_, a in tr])
            for r, m, a in te: errs.append(abs(predict(c, r, m, interaction) - a))
        return np.mean(errs) if errs else float('nan')
    ri, rs_ = loro(True), loro(False)

    print(f"  LOCO (leave-cell):   interaction={li:.4f}  separable-null={ls:.4f}  -> {'interaction WINS' if li<ls else 'null wins/ties'}")
    print(f"  LORO (leave-r/extrapolate compute): interaction={ri:.4f}  separable-null={rs_:.4f}  -> {'interaction WINS' if ri<rs_ else 'null wins/ties'}")
    print(f"  verdict: closed-form beta1={beta1:+.3f}"
          + ("  (held-out supported)" if (li < ls) else "  (NOT held-out supported)"))
print("\nDONE  (compare beta1 here to the free-intercept 'cheating' fit: MATH-0.5B was -0.26, MATH-1.5B -0.08)")
