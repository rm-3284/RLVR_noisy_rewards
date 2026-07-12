"""SETTLED MATH-1.5B acc(r,m) β1 fit (2026-07-06), corner now out of queue.
Addresses the timeout-cap: r128 row caps at ~step 200 while r8/r16 reach ~234, so reading each cell at its own
final step is apples-to-oranges. Report β1 under 4 defensible reads → a RANGE + DIRECTION, never one magnitude.
Reads: (A) single-last-val, (B) mean-last-5, (C) common-step (truncate all cells to the min final step), (D) common-step mean-last-5.
Per-seed convergence gate: >=10 vals, non-collapse (acc>=0.2), >=4 seeds/cell. Fit logit(acc)=a(r)-β(1-m)-β1(1-m)log2 r; LOCO.
"""
import glob, math
import numpy as np
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
def keyof(it): return it.key or ("/".join(it.nested_key) if it.nested_key else "")
def series(f):
    ds=DataStore(); ds.open_for_scan(f); out=[]
    while True:
        try: rb=ds.scan_data()
        except Exception: break
        if rb is None: break
        r=pb.Record()
        try: r.ParseFromString(rb)
        except Exception: continue
        if r.WhichOneof("record_type")=="history":
            d={keyof(i):i.value_json for i in r.history.item}
            if "validation/accuracy" in d:
                try: out.append((int(d.get("_step","-1")), float(d["validation/accuracy"])))
                except: pass
    out.sort(); return out

CELLS={8:{0.0:"r8-fp0.0-fn0.0",0.15:"r8-fp0.15-fn0.15",0.3:"r8-fp0.3-fn0.3"},
       16:{0.0:"r16-fp0.0-fn0.0",0.15:"r16-fp0.15-fn0.15",0.3:"r16-fp0.3-fn0.3"},
       128:{0.0:"r128-fp0.0-fn0.0",0.15:"r128-fp0.15-fn0.15",0.3:"r128-fp0.3-fn0.3"}}
def cell_dirs(tag): return sorted(glob.glob(f"logs/{'tsMb' if tag.startswith('r16') else 'tsM'}-1.5B-{tag}-s*"))

# collect per-seed series
raw={}  # (r,p) -> list of series
for r,pm in CELLS.items():
    for p,tag in pm.items():
        ss=[]
        for d in cell_dirs(tag):
            fs=glob.glob(f"{d}/exp_001/wandb/wandb/offline-run-*/run-*.wandb")
            if fs:
                s=series(fs[0])
                if s: ss.append(s)
        raw[(r,p)]=ss

# common step = min over ALL seeds of that seed's final step
COMMON=min(s[-1][0] for ss in raw.values() for s in ss if s)
print(f"common step (min final across all seeds) = {COMMON}")

def read_seed(s, mode):
    if mode=="A": return s[-1][1]
    if mode=="B": return float(np.mean([a for _,a in s[-5:]]))
    trunc=[(st,a) for st,a in s if st<=COMMON] or s[:1]
    if mode=="C": return trunc[-1][1]
    if mode=="D": return float(np.mean([a for _,a in trunc[-5:]]))

def cellmeans(mode):
    cm={}
    for (r,p),ss in raw.items():
        vals=[]
        for s in ss:
            nv=len(s)
            a=read_seed(s,mode)
            if nv>=10 and a>=0.2: vals.append(a)   # convergence gate
        if len(vals)>=4: cm[(r,p)]=(1-2*p, float(np.mean(vals)), len(vals))
    return cm

def logit(a): a=min(max(a,1e-4),1-1e-4); return math.log(a/(1-a))
def fit(cm):
    rlist=sorted({r for (r,_) in cm}); ridx={r:i for i,r in enumerate(rlist)}
    X=[];y=[]
    for (r,p),(m,acc,n) in cm.items():
        row=[0.0]*len(rlist); row[ridx[r]]=1.0; row+=[-(1-m),-(1-m)*math.log2(r)]
        X.append(row); y.append(logit(acc))
    X=np.array(X); y=np.array(y)
    c,*_=np.linalg.lstsq(X,y,rcond=None)
    # LOCO
    errs=[]
    accs=[cm[k][1] for k in cm]
    keys=list(cm)
    for i in range(len(keys)):
        idx=[j for j in range(len(keys)) if j!=i]
        ci,*_=np.linalg.lstsq(X[idx],y[idx],rcond=None)
        pi=1/(1+math.exp(-(X[i]@ci))); errs.append(abs(pi-accs[i]))
    return c[-2], c[-1], float(np.mean(errs)), len(keys)

print(f"\n{'read method':<34} {'ncells':>6} {'beta':>7} {'beta1':>8} {'LOCO':>7}  verdict")
b1s=[]
for mode,label in [("A","A single-last-val"),("B","B mean-last-5"),
                   ("C","C common-step single"),("D","D common-step mean-last-5")]:
    cm=cellmeans(mode)
    b,b1,loco,n=fit(cm); b1s.append(b1)
    print(f"{label:<34} {n:>6} {b:>+7.2f} {b1:>+8.3f} {loco:>7.4f}  {'interaction' if b1<-0.05 else 'separable/~0'}")
print(f"\nβ1 RANGE across reads: [{min(b1s):+.3f}, {max(b1s):+.3f}]  (spread {max(b1s)-min(b1s):.3f})")
print(f"DIRECTION: {'all interaction-side' if all(b<0 for b in b1s) else 'MIXED sign'}; "
      f"{'all |β1|>0.05 (robust mild interaction)' if all(abs(b)>0.05 for b in b1s) else 'some |β1|<0.05 → near-zero/undecided'}")
