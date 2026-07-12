"""Two-variable functional-form comparison: does accuracy depend on fp,fn ONLY through the
margin m=1-fp-fn (collapse), or on fp,fn SEPARATELY, and is any asymmetry base-rate-modulated?

Pools multiple configs (each with its own base rate b = clean acc). Fits, on the logit scale,
anchored so acc(0,0)=b_c:
  M1 collapse (symmetric):        logit(acc) = logit(b) - βm*(fp+fn)
  M2 separable (const asymmetry): logit(acc) = logit(b) - βm*(fp+fn) - βa*(fp-fn)
  M3 base-rate-modulated asym:    logit(acc) = logit(b) - βm*(fp+fn) - βa*(fp-fn)*(1-2b)
Collapse <=> βa=0. M3 says FP gets costlier as base rate falls (precision mechanism).
Model-select by LEAVE-ONE-CONFIG-OUT held-out MAE (extrapolation), not in-sample R².
Usage: python jobscripts/fit_forms.py  (writes logs/forms_fit.txt)
"""
import re, statistics as st
from collections import defaultdict
import numpy as np
import wandb
api = wandb.Api()

# config -> wandb name regex ; base rate computed from that config's (0,0) cell
CONFIGS = {
    "gsm8k-1.5B": r"^c1-1.5B-fp",
    "math-1.5B":  r"^pMATH-1.5B-fp",
    "math-3B":    r"^pMATH-3B-fp",
    "gsm8k-3B":   r"^p3B-gsm8k-fp",
    "gsm8k-0.5B": r"^p05B-gsm8k-fp",
    "olmo-1B":    r"^olmo-gsm8k-fp",
}
NR = re.compile(r"-r(\d+)-s\d+$")

def pull(regex):
    best = {}
    for r in api.runs("rm4411-princeton-university/RLVR",
                      filters={"display_name": {"$regex": regex}}, per_page=500):
        em = r.config.get("env", {}).get("math", {}); fp, fn = em.get("fp"), em.get("fn")
        acc = r.summary.get("validation/accuracy")
        if fp is None or acc is None: continue
        step = r.summary.get("_step", 0) or 0
        # dedup by name keep max step; drop obviously-truncated (<60% of this cfg's max step later)
        key = r.name
        if key not in best or step > best[key][0]:
            best[key] = (step, round(float(fp), 2), round(float(fn), 2), float(acc))
    if not best: return {}
    maxstep = max(v[0] for v in best.values())
    cells = defaultdict(list)
    for step, fp, fn, acc in best.values():
        if step < 0.6 * maxstep: continue
        cells[(fp, fn)].append(acc)
    return {k: st.mean(v) for k, v in cells.items()}

data = {}  # config -> {(fp,fn): acc}, base rate
for name, rgx in CONFIGS.items():
    c = pull(rgx)
    if (0.0, 0.0) in c and len(c) >= 3:
        data[name] = c
    else:
        print(f"skip {name}: cells={len(c)} clean={'y' if (0.0,0.0) in c else 'n'}")

def logit(p): return np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))

# build design rows: (config, fp, fn, acc, b)
rows = []
for cfg, cells in data.items():
    b = cells[(0.0, 0.0)]
    for (fp, fn), acc in cells.items():
        rows.append((cfg, fp, fn, acc, b))
print(f"\nconfigs: " + ", ".join(f"{k}(b={data[k][(0.0,0.0)]:.2f},n={len(data[k])})" for k in data))
print(f"total cells: {len(rows)}")

def design(rows, model):
    X, y = [], []
    for cfg, fp, fn, acc, b in rows:
        tgt = logit(acc) - logit(b)  # anchored: 0 at clean
        if model == "M1":   feats = [-(fp + fn)]
        elif model == "M2": feats = [-(fp + fn), -(fp - fn)]
        elif model == "M3": feats = [-(fp + fn), -(fp - fn) * (1 - 2 * b)]
        X.append(feats); y.append(tgt)
    return np.array(X), np.array(y)

def fit(X, y):  # least squares, no intercept (anchored)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); return coef

def predict(rows, model, coef):
    preds = []
    for cfg, fp, fn, acc, b in rows:
        if model == "M1":   f = np.array([-(fp + fn)])
        elif model == "M2": f = np.array([-(fp + fn), -(fp - fn)])
        elif model == "M3": f = np.array([-(fp + fn), -(fp - fn) * (1 - 2 * b)])
        lg = logit(b) + f @ coef
        preds.append(1 / (1 + np.exp(-lg)))
    return np.array(preds)

cfgs = list(data)
per = {m: {} for m in ("M1", "M2", "M3")}
print(f"\n{'model':6} {'in-MAE':>8} {'LOCO-MAE':>9}  coefs")
for model in ("M1", "M2", "M3"):
    X, y = design(rows, model); coef = fit(X, y)
    pin = predict(rows, model, coef); acc_true = np.array([r[3] for r in rows])
    in_mae = np.mean(np.abs(pin - acc_true))
    errs = []
    for held in cfgs:
        tr = [r for r in rows if r[0] != held]; te = [r for r in rows if r[0] == held]
        if not te: continue
        Xt, yt = design(tr, model); c = fit(Xt, yt)
        pte = predict(te, model, c); yte = np.array([r[3] for r in te])
        e = np.mean(np.abs(pte - yte)); errs.append(e); per[model][held] = e
    cs = ", ".join(f"{v:+.2f}" for v in coef)
    print(f"{model:6} {in_mae:8.3f} {np.mean(errs):9.3f}  [{cs}]")

print(f"\nPER-CONFIG held-out MAE (the asymmetry lives in low-base-rate configs):")
print(f"  {'config':12} {'b':>4}  {'M1':>6} {'M2':>6} {'M3':>6}")
for held in sorted(cfgs, key=lambda c: data[c][(0.0,0.0)]):
    b = data[held][(0.0,0.0)]
    print(f"  {held:12} {b:4.2f}  " + " ".join(f"{per[m].get(held,float('nan')):6.3f}" for m in ("M1","M2","M3")))

print(f"\nMatched-margin asymmetry check (actual FP vs FN at same m, per config):")
for cfg in cfgs:
    cells = data[cfg]
    for (fp,fn) in list(cells):
        mir=(fn,fp)
        if fp>fn and mir in cells:
            print(f"  {cfg:12} FP{(fp,fn)}={cells[(fp,fn)]:.3f}  FN{mir}={cells[mir]:.3f}  diff={cells[(fp,fn)]-cells[mir]:+.3f}")

print("\nInterpretation: M1=collapse(symmetric), M2=const asymmetry, M3=base-rate-modulated asymmetry.")
print("If M3 LOCO-MAE < M2 < M1 => asymmetry is real AND base-rate-modulated (precision mechanism).")
print("βa>0 in M2/M3 => FP costlier than FN. In M3, cost scales with (1-2b): FP-worse at low base rate.")
print("DONE")
