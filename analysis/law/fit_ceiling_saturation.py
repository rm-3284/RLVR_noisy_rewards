"""Does NOISE lower the COMPUTE CEILING? (the strong "you're fucked no matter what" test)

Fit saturating  acc(r) = A_inf - c * r^-alpha  per arm (clean m=1 vs noisy m=0.6) and compare the
asymptote A_inf. If A_inf(noisy) < A_inf(clean) with a CI clear of 0 -> even infinite compute cannot
reach clean performance = noise caps the ceiling. If the gap SHRINKS to ~0 with r -> compute closes it.

Uses GSM8K-1.5B, which has the widest compute axis on disk: r = 4,8,16,32,64,128,256.
Offline .wandb read, dedup by seed, >=10-eval gate, mean-last-5. Seed-bootstrap CI on the asymptote gap.
"""
import glob, os, re
import numpy as np
from scipy.optimize import curve_fit
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

def keyof(it): return it.key or ("/".join(it.nested_key) if it.nested_key else "")
def newest(d):
    r = glob.glob(f"{d}/exp_*/wandb/wandb/offline-run-*/run-*.wandb")
    return max(r, key=os.path.getmtime) if r else None
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

PREFIXES = ["c2-gsm8k-1.5B", "c2b-gsm8k-1.5B", "c2c-gsm8k-1.5B", "c1r8-1.5B", "c1-1.5B"]
RS = [4, 8, 16, 32, 64, 128, 256]
ARMS = {"clean (m=1.0)": 0.0, "noisy (m=0.6)": 0.2}

def per_seed(fp, r):
    seen = {}
    for pre in PREFIXES:
        for d in glob.glob(f"logs/{pre}-fp{fp}-fn{fp}-r{r}-s*"):
            f = newest(d)
            if not f: continue
            s = series(f)
            if len(s) >= 10:
                sm = re.search(r'-s(\d+)$', d); sid = sm.group(1) if sm else d
                seen.setdefault(sid, []).append(float(np.mean([a for _, a in s[-5:]])))
    return [float(np.mean(v)) for v in seen.values()]

# collect per-seed values per (arm, r)
data = {name: {r: per_seed(fp, r) for r in RS} for name, fp in ARMS.items()}

def sat(r, A, c, al): return A - c * np.power(r, -al)
def fit_arm(rs, ys):
    p0 = [max(ys), max(0.05, max(ys) - min(ys)), 0.5]
    popt, _ = curve_fit(sat, np.array(rs, float), np.array(ys, float),
                        p0=p0, bounds=([0, 0, 0.05], [1, 2, 3]), maxfev=40000)
    pred = sat(np.array(rs, float), *popt)
    r2 = 1 - np.sum((np.array(ys) - pred) ** 2) / (np.var(ys) * len(ys) + 1e-12)
    return popt, r2

print("GSM8K-1.5B  saturating fit  acc(r)=A_inf - c*r^-alpha   (per arm)\n")
Ainf = {}
for name in ARMS:
    rs = [r for r in RS if data[name][r]]
    ys = [float(np.mean(data[name][r])) for r in rs]
    (A, c, al), r2 = fit_arm(rs, ys)
    Ainf[name] = A
    print(f"{name:16}  points " + " ".join(f"r{r}:{y:.3f}" for r, y in zip(rs, ys)))
    print(f"{'':16}  A_inf={A:.3f}  c={c:.3f}  alpha={al:.2f}  R2={r2:.3f}\n")

gap = Ainf["clean (m=1.0)"] - Ainf["noisy (m=0.6)"]
print(f"ASYMPTOTE gap  A_inf(clean) - A_inf(noisy) = {gap:+.3f}")

# ---- seed-bootstrap CI on the asymptote gap ----
rng = np.random.default_rng(0)
gaps = []
for _ in range(1000):
    A = {}
    ok = True
    for name in ARMS:
        rs, ys = [], []
        for r in RS:
            v = data[name][r]
            if not v: continue
            rs.append(r); ys.append(float(np.mean(rng.choice(v, size=len(v), replace=True))))
        try:
            (a, _, _), _ = fit_arm(rs, ys); A[name] = a
        except Exception:
            ok = False; break
    if ok: gaps.append(A["clean (m=1.0)"] - A["noisy (m=0.6)"])
gaps = np.array(gaps)
lo, hi = np.percentile(gaps, [2.5, 97.5])
frac_pos = float(np.mean(gaps > 0))
print(f"  bootstrap 95% CI = [{lo:+.3f}, {hi:+.3f}]   P(gap>0) = {frac_pos:.2f}   (n={len(gaps)} resamples)")

# ---- does the finite-r gap shrink with compute? ----
print("\nfinite-r gap (clean - noisy) at each r  (shrinking->compute closes it; flat->permanent):")
for r in RS:
    c_, n_ = data["clean (m=1.0)"][r], data["noisy (m=0.6)"][r]
    if c_ and n_:
        print(f"  r={r:>3}: {np.mean(c_)-np.mean(n_):+.3f}")

verdict = ("NOISE LOWERS THE CEILING (gap CI clear of 0) -> can't compute your way to clean"
           if lo > 0 else
           "ceiling gap NOT clearly > 0 -> can't claim a permanent asymptote deficit from this")
print(f"\nVERDICT: {verdict}")
print("DONE")
