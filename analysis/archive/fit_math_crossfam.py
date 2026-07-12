"""Fit acc(r,m) tradeoff form on the now-complete MATH grids (2026-07-07).
logit(acc) = a(r) - beta*(1-m) - beta1*(1-m)*log2(r) ; m = 1 - fp - fn (symmetric: m=1-2fp).
beta1<0 = INTERACTION (compute buys back noise); beta1~0 = SEPARABLE.
Read: per-seed mean-last-5, cell = mean over converged seeds (>=10 vals). LOCO leave-one-cell-out MAE.
Report beta1 DIRECTION + magnitude, honest about r128 timeout-cap (read also at a common step)."""
import glob, os, math, re
import numpy as np
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
def keyof(it): return it.key or ("/".join(it.nested_key) if it.nested_key else "")
def newest(d):
    r=glob.glob(f"{d}/exp_*/wandb/wandb/offline-run-*/run-*.wandb"); return max(r,key=os.path.getmtime) if r else None
def series(f):
    ds=DataStore(); ds.open_for_scan(f); out=[]
    while True:
        try: rb=ds.scan_data()
        except: break
        if rb is None: break
        r=pb.Record()
        try: r.ParseFromString(rb)
        except: continue
        if r.WhichOneof("record_type")=="history":
            d={keyof(i):i.value_json for i in r.history.item}
            if "validation/accuracy" in d:
                try: out.append((int(d.get("_step","-1")),float(d["validation/accuracy"])))
                except: pass
    out.sort(); return out

# model -> list of (r, fp, glob-patterns) ; MATH symmetric noise fp==fn
MODELS={
 "MATH-0.5B": ("logs/ts05M-0.5B-{r}-fp{fp}-fn{fp}-s*","logs/aiMATH05hi-0.5B-{r}-fp{fp}-fn{fp}-s*"),
 "MATH-1.5B": ("logs/tsM-1.5B-{r}-fp{fp}-fn{fp}-s*","logs/tsMb-1.5B-{r}-fp{fp}-fn{fp}-s*"),
 # OLMo-1B cross-family point: use the LIVE re-run (v4o1-*), NOT the dead 2026-07-07 olmomath-* set
 # (its r128 crashed at step ~40). v4o1 has fp=0 asymmetric column, so the diagonal m=1-2fp
 # assumption in _design must be generalized to m=1-fp-fn before enabling this — TODO, not wired yet.
}
RS={"MATH-0.5B":[8,16,32,64],"MATH-1.5B":[8,16,32,64,128]}
FPS=[0.0,0.15,0.3]
def cellmean(model,r,fp):
    # DEDUP by seed: ts05M + aiMATH05hi share seeds 1-5 at r64 (same config, 2 clusters).
    # Counting both = fake n=10. Collect per-seed, average duplicate cluster-runs -> one value per unique seed.
    per_seed={}
    for pat in MODELS[model]:
        for d in glob.glob(pat.format(r=f"r{r}",fp=fp)):
            f=newest(d)
            if not f: continue
            s=series(f)
            if len(s)>=10:  # convergence gate
                sm=re.search(r'-s(\d+)$', d); sid=sm.group(1) if sm else d
                per_seed.setdefault(sid,[]).append(float(np.mean([a for _,a in s[-5:]])))
    return [float(np.mean(v)) for v in per_seed.values()]  # one value per unique seed
def logit(a): a=min(max(a,1e-4),1-1e-4); return math.log(a/(1-a))
def _design(cells, interaction=True):
    rlist=sorted({r for (r,_) in cells}); ridx={r:i for i,r in enumerate(rlist)}
    X=[]; y=[]; keys=list(cells)
    for (r,fp) in keys:
        m=1-2*fp
        row=[0.0]*len(rlist); row[ridx[r]]=1.0; row.append(-(1-m))
        if interaction: row.append(-(1-m)*math.log2(r))
        X.append(row); y.append(logit(cells[(r,fp)][0]))
    return np.array(X), np.array(y), keys
def _loco(X,y,keys,cells):
    errs=[]
    for i in range(len(keys)):
        idx=[j for j in range(len(keys)) if j!=i]
        ci,*_=np.linalg.lstsq(X[idx],y[idx],rcond=None)
        pi=1/(1+math.exp(-(X[i]@ci))); errs.append(abs(pi-cells[keys[i]][0]))
    return float(np.mean(errs))
def fit(cells):
    Xi,yi,keys=_design(cells,True); ci,*_=np.linalg.lstsq(Xi,yi,rcond=None); loco_i=_loco(Xi,yi,keys,cells)
    Xs,ys,_=_design(cells,False); loco_s=_loco(Xs,ys,keys,cells)  # separable null (beta1=0)
    return ci[-2], ci[-1], loco_i, loco_s, len(keys)

for model in MODELS:
    cells={}; grid_ok=True
    print(f"\n=== {model} ===")
    for r in RS[model]:
        row=[]
        for fp in FPS:
            sd=cellmean(model,r,fp);
            if len(sd)>=4: cells[(r,fp)]=(float(np.mean(sd)),len(sd))
            row.append(f"r{r}m{1-2*fp:.1f}:{np.mean(sd):.3f}(n{len(sd)})" if sd else f"r{r}m{1-2*fp:.1f}:--")
        print("  "+"  ".join(row))
    ncells=len(cells); fittable=sum(1 for (r,_) in cells if r>=64)>0 and sum(1 for (r,_) in cells if r<=16)>0
    if ncells>=6 and fittable:
        b,b1,loco_i,loco_s,n=fit(cells)
        beats = loco_i < loco_s   # does interaction beat separable on held-out?
        verdict = ("INTERACTION real" if (b1<-0.05 and beats) else
                   "interaction NOT held-out-supported (separable null ties/wins)" if b1<-0.05 else "SEPARABLE (~0)")
        print(f"  FIT: beta={b:+.3f} beta1={b1:+.3f} | LOCO interaction={loco_i:.4f} vs separable-null={loco_s:.4f} "
              f"({'interaction WINS' if beats else 'null wins/ties'}) | ncells={n} -> {verdict}")
    else:
        print(f"  NOT fittable yet (ncells={ncells}, need >=6 spanning r<=16 AND r>=64)")
