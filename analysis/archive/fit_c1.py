"""Fit the C1 GSM8K-1.5B surface: test whether accuracy collapses onto the margin m=1-fp-fn,
quantify any residual asymmetry vs (fp-fn), and report held-out R2. Uses both acc_final and acc_best.
Writes logs/c1_fit.txt. Usage: python jobscripts/fit_c1.py
"""
import os, numpy as np
import wandb
ENTITY = os.environ.get("WANDB_ENTITY", "rm4411-princeton-university")

def pull():
    api = wandb.Api()
    rows = []
    for r in api.runs(f"{ENTITY}/RLVR"):
        if not r.name.startswith("c1-1.5B"): continue
        em = r.config.get("env", {}).get("math", {})
        fp, fn = em.get("fp"), em.get("fn")
        acc = r.summary.get("validation/accuracy")
        if fp is None or acc is None: continue
        best = None
        try:
            h = r.history(keys=["validation/accuracy"], samples=300, pandas=False)
            vals = [x.get("validation/accuracy") for x in h if x.get("validation/accuracy") is not None]
            best = max(vals) if vals else acc
        except Exception:
            best = acc
        rows.append((float(fp), float(fn), float(acc), float(best)))
    return np.array(rows)  # cols: fp, fn, acc_final, acc_best

def fit_metric(D, col, name, out):
    fp, fn, y = D[:,0], D[:,1], D[:,col]
    m = 1 - fp - fn
    out.append(f"\n=== {name}  (n={len(y)} runs) ===")
    # linear + quadratic in m
    for deg, lbl in [(1,"linear a+b*m"), (2,"quad a+b*m+c*m^2")]:
        X = np.vander(m, deg+1)  # highest power first
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss_res = ((y-pred)**2).sum(); ss_tot = ((y-y.mean())**2).sum()
        r2 = 1 - ss_res/ss_tot
        rmse = np.sqrt((( y-pred)**2).mean())
        out.append(f"  {lbl}: R2(in-sample)={r2:.3f} RMSE={rmse:.4f} coef={np.round(coef,4).tolist()}")
    # held-out: leave-one-fp-out (extrapolation across the fp axis)
    r2s = []
    for fp_hold in np.unique(fp):
        tr = fp != fp_hold; te = fp == fp_hold
        if te.sum()==0 or tr.sum()<3: continue
        X = np.vander(m[tr],3); coef,*_=np.linalg.lstsq(X,y[tr],rcond=None)
        pe = np.vander(m[te],3)@coef
        r2s.append(1 - ((y[te]-pe)**2).sum()/max(((y[te]-y[te].mean())**2).sum(),1e-9))
    if r2s: out.append(f"  held-out (leave-one-fp-out) mean R2 = {np.mean(r2s):.3f}")
    # residual asymmetry: resid = y - quad(m), regress on (fp-fn); bootstrap slope CI
    X = np.vander(m,3); coef,*_=np.linalg.lstsq(X,y,rcond=None); resid = y - X@coef
    d = fp - fn
    slope = np.polyfit(d, resid, 1)[0]
    bs = []
    idx = np.arange(len(y))
    for _ in range(2000):
        s = np.random.choice(idx, len(idx), replace=True)
        bs.append(np.polyfit(d[s], resid[s], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    excl0 = "EXCLUDES 0 (real asymmetry)" if (lo>0 or hi<0) else "includes 0 (symmetric)"
    out.append(f"  residual-vs-(fp-fn) slope = {slope:+.4f}  95%CI [{lo:+.4f},{hi:+.4f}]  -> {excl0}")

def main():
    D = pull()
    out = [f"C1 fit — GSM8K 1.5B r=32 — {len(D)} runs"]
    # cell coverage
    cells = {}
    for fp,fn,_,_ in D: cells[(round(fp,2),round(fn,2))] = cells.get((round(fp,2),round(fn,2)),0)+1
    out.append(f"cells: {len(cells)}  seeds/cell: min={min(cells.values())} max={max(cells.values())}")
    fit_metric(D, 2, "acc_final vs margin m", out)
    fit_metric(D, 3, "acc_best  vs margin m", out)
    txt = "\n".join(out)
    print(txt)
    open("logs/c1_fit.txt","w").write(txt+"\n")

if __name__ == "__main__":
    main()

# --- candidate functional forms (model selection by held-out R2) ---
def fit_forms(D, col, name, out):
    from scipy.optimize import curve_fit
    fp, fn, y = D[:,0], D[:,1], D[:,col]
    m = 1 - fp - fn
    forms = {
        "linear":      (lambda m,a,b: a+b*m,                      [0.2,0.5]),
        "quadratic":   (lambda m,a,b,c: a+b*m+c*m*m,              [0.2,0.5,0.1]),
        "logistic":    (lambda m,L,k,m0,F: F+(L-F)/(1+np.exp(-k*(m-m0))), [0.76,6,0.4,0.2]),
        "exp_sat":     (lambda m,F,C,k: F+(C-F)*(1-np.exp(-k*m)), [0.2,0.76,3.0]),
    }
    out.append(f"\n--- {name}: candidate forms (held-out = leave-one-fp-out) ---")
    fps = np.unique(fp)
    for nm,(f,p0) in forms.items():
        try:
            popt,_ = curve_fit(f, m, y, p0=p0, maxfev=20000)
            pred = f(m,*popt); r2 = 1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
            ho=[]
            for h in fps:
                tr=fp!=h; te=fp==h
                if te.sum()==0 or tr.sum()<len(p0)+1: continue
                try:
                    pp,_=curve_fit(f,m[tr],y[tr],p0=p0,maxfev=20000)
                    pe=f(m[te],*pp); ho.append(1-((y[te]-pe)**2).sum()/max(((y[te]-y[te].mean())**2).sum(),1e-9))
                except Exception: pass
            hor = np.mean(ho) if ho else float('nan')
            out.append(f"  {nm:10s} R2_in={r2:.3f}  R2_heldout={hor:.3f}  params={np.round(popt,3).tolist()}")
        except Exception as e:
            out.append(f"  {nm:10s} FIT FAILED ({e})")

if __name__ == "__main__":
    # append form comparison to the report
    D = pull()
    out = ["\n\n########## FUNCTIONAL FORM SELECTION ##########"]
    fit_forms(D, 2, "acc_final", out)
    fit_forms(D, 3, "acc_best", out)
    txt = "\n".join(out); print(txt)
    open("logs/c1_fit.txt","a").write(txt+"\n")
