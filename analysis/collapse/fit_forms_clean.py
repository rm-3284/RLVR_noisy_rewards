"""CLEAN margin-collapse fit — converged-9 basis only (2026-07-08 skimper #19 correction).

Excludes the frozen/undertrained configs that contaminated the old "~12 config / LOCO 0.036"
number: p7B-GSM8K (frozen at val_at_start, all cells [1] real eval), pMATH-1.5B (~15-step
snapshot, [3] real), pMATH-3B ([2-3] real), pMATHl12 ([7] real). Converged MATH-1.5B margin
behavior is instead carried by mL4 [34] + mL5 [48] real evals, so pMATH-1.5B is redundant.

Same M1/M2/M3 leave-one-config-out logic as fit_forms.py.
"""
import re, statistics as st
from collections import defaultdict
import numpy as np
import wandb
api = wandb.Api()

# 9 CONVERGED configs (>=10 real evals each, verified via offline ÷2 recount)
CONFIGS = {
    "gsm8k-1.5B": r"^c1-1.5B-fp",
    "gsm8k-0.5B": r"^p05B-gsm8k-fp",
    "gsm8k-3B":   r"^p3B-gsm8k-fp",
    "math-0.5B":  r"^pMATH-0.5B-fp",
    "olmo-gsm8k": r"^olmo-gsm8k-fp",
    "olmo-math":  r"^olmomath-fp",
    "ext-1.5B":   r"^ext-gsm8k-1.5B-fp",
    "math-1.5B-mL4": r"^mL4-1.5B-fp",
    "math-1.5B-mL5": r"^mL5-1.5B-fp",
}

def pull(regex):
    best = {}
    for r in api.runs("rm4411-princeton-university/RLVR",
                      filters={"display_name": {"$regex": regex}}, per_page=500):
        em = r.config.get("env", {}).get("math", {}); fp, fn = em.get("fp"), em.get("fn")
        acc = r.summary.get("validation/accuracy")
        if fp is None or acc is None: continue
        step = r.summary.get("_step", 0) or 0
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

data = {}
for name, rgx in CONFIGS.items():
    c = pull(rgx)
    if (0.0, 0.0) in c and len(c) >= 3:
        data[name] = c
    else:
        print(f"SKIP {name}: cells={len(c)} clean={'y' if (0.0,0.0) in c else 'n'}")

def logit(p): return np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))

rows = []
for cfg, cells in data.items():
    b = cells[(0.0, 0.0)]
    for (fp, fn), acc in cells.items():
        rows.append((cfg, fp, fn, acc, b))
print(f"\nCONVERGED configs ({len(data)}): " + ", ".join(f"{k}(b={data[k][(0.0,0.0)]:.2f},n={len(data[k])})" for k in data))
print(f"total cells: {len(rows)}")

def design(rows, model):
    X, y = [], []
    for cfg, fp, fn, acc, b in rows:
        tgt = logit(acc) - logit(b)
        if model == "M1":   feats = [-(fp + fn)]
        elif model == "M2": feats = [-(fp + fn), -(fp - fn)]
        elif model == "M3": feats = [-(fp + fn), -(fp - fn) * (1 - 2 * b)]
        X.append(feats); y.append(tgt)
    return np.array(X), np.array(y)

def fit(X, y):
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

print(f"\nPER-CONFIG held-out MAE (sorted by base rate):")
print(f"  {'config':16} {'b':>4}  {'M1':>6} {'M2':>6} {'M3':>6}")
for held in sorted(cfgs, key=lambda c: data[c][(0.0,0.0)]):
    b = data[held][(0.0,0.0)]
    print(f"  {held:16} {b:4.2f}  " + " ".join(f"{per[m].get(held,float('nan')):6.3f}" for m in ("M1","M2","M3")))
print("\nM1=collapse. Collapse holds iff M1 LOCO <= M2/M3 (asymmetry does not improve extrapolation).")
print("DONE")
