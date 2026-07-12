"""Gated per-config acc(r,m) fit for MATH-1.5B, now that it has 3 compute points (r8,r16,r128).

⚠️ METHODOLOGY WARNING (2026-07-06, skimper catch): this script reads SINGLE-LAST-VAL per seed. The r128 row is
TIMEOUT-CAPPED (~19-22 vals @ ~step 200) while r8/r16 reach 25 vals @ step 234 — so single-last-val compares cells
at DIFFERENT training steps, and β1's MAGNITUDE swings −0.069..−0.137 by read choice (spread > LOCO error). Before the
FINAL settled fit: read ALL cells at a COMMON STEP (truncate r8/r16 to the r128 timeout step) AND/OR average the
plateau (mean-last-5), report β1 under multiple reads as a RANGE + DIRECTION (mild interaction-side, |β1|≈0.07-0.14),
never a fixed magnitude. See TRADEOFF_form.md "MATH-1.5B β1 — PENDING / METHODOLOGY FIX REQUIRED".

Extracts final converged validation/accuracy from OFFLINE wandb, enforces GATE-A (n=5, >=10 vals),
GATE-B (>=3 r spanning low<=16 & high>=64, non-saturated), then fits
   logit(acc) = a(r) - beta*(1-m) - beta1*(1-m)*log2(r)     (per-r intercepts a(r))
and reports leave-one-cell-out held-out MAE. Does NOT touch fig7 (money plot stays PENDING on endpoints).
"""
import glob, math
import numpy as np
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

def keyof(it): return it.key or ("/".join(it.nested_key) if it.nested_key else "")

def val_series(f):
    ds = DataStore(); ds.open_for_scan(f); out=[]
    while True:
        try: rb = ds.scan_data()
        except Exception: break
        if rb is None: break
        r = pb.Record()
        try: r.ParseFromString(rb)
        except Exception: continue
        if r.WhichOneof("record_type")=="history":
            d={keyof(i):i.value_json for i in r.history.item}
            if "validation/accuracy" in d:
                try: out.append((int(d.get("_step","-1")), float(d["validation/accuracy"])))
                except: pass
    out.sort(); return out

# MATH symmetric noise p -> margin m = 1-2p
CELLS = {8:  {0.0:"r8-fp0.0-fn0.0", 0.15:"r8-fp0.15-fn0.15", 0.3:"r8-fp0.3-fn0.3"},
         16: {0.0:"r16-fp0.0-fn0.0",0.15:"r16-fp0.15-fn0.15",0.3:"r16-fp0.3-fn0.3"},
         128:{0.0:"r128-fp0.0-fn0.0",0.15:"r128-fp0.15-fn0.15",0.3:"r128-fp0.3-fn0.3"}}
MINVALS = 10  # GATE-A: MATH convergence bar

def cell_dirs(tag):
    # tsMb holds r16; tsM holds r8/r128
    pref = "tsMb" if tag.startswith("r16") else "tsM"
    return sorted(glob.glob(f"logs/{pref}-1.5B-{tag}-s*"))

rows=[]  # (r, p, m, mean_acc_over_converged, n_converged, n_total)
print(f"{'cell':<22} {'conv/tot':>8} {'mean':>7}   per-seed (acc,nvals)")
for r,pm in CELLS.items():
    for p,tag in pm.items():
        seeds=[]  # (acc, nval)
        for d in cell_dirs(tag):
            fs=glob.glob(f"{d}/exp_001/wandb/wandb/offline-run-*/run-*.wandb")
            if not fs: continue
            s=val_series(fs[0])
            if s: seeds.append((s[-1][1], len(s)))
        if not seeds:
            print(f"{tag:<22} {'0/0':>8}  NO DATA"); continue
        # per-seed convergence: keep seeds with >=MINVALS vals AND non-collapse (acc>=0.2, collapse-robust)
        conv=[a for a,nvl in seeds if nvl>=MINVALS and a>=0.2]
        m=1-2*p
        rows.append((r,p,m,float(np.mean(conv)) if conv else None,len(conv),len(seeds)))
        mean=f"{np.mean(conv):.3f}" if conv else "  --"
        print(f"{tag:<22} {f'{len(conv)}/{len(seeds)}':>8} {mean:>7}   {[(round(a,3),nvl) for a,nvl in seeds]}")

# GATE-A enforcement: each cell needs >=4 CONVERGED seeds (>=MINVALS vals, non-collapse)
print("\n=== GATE-A check (need >=4 converged seeds/cell: >=%d vals, non-collapse) ===" % MINVALS)
good=[x for x in rows if x[4]>=4 and x[3] is not None]
bad =[x for x in rows if not (x[4]>=4 and x[3] is not None)]
for x in bad: print(f"  EXCLUDED r{x[0]} p{x[1]}: only {x[4]} converged of {x[5]} seeds")
rset=sorted(set(x[0] for x in good));
print(f"  usable cells: {len(good)} across r={rset}")
if len(rset)<3 or max(rset)<64 or min(rset)>16:
    print("  !! GATE-B FAIL: need >=3 r spanning low<=16 & high>=64. STOP, not fittable yet."); raise SystemExit

# ANTI-ARTIFACT GUARD (2026-07-06, added after retraction): the highest-r highest-noise corner
# (r128 x m=0.4) is the SOLE decider of beta1's sign. If GATE-A excluded it (under-converged), beta1 is
# UNDEFINED and fitting the remaining 8 cells reproduces the RETRACTED artifact (-0.10). Refuse to fit.
rmax=max(rset)
corner_present = any((x[0]==rmax and abs(x[1]-0.3)<1e-9) for x in good)
if not corner_present:
    print(f"\n  !! ANTI-ARTIFACT STOP: the r{rmax}xp0.3 corner (the β1 decider) is under-converged and was excluded.")
    print("     β1 is UNDEFINED without it — the retracted -0.10 WAS this exact artifact (see TRADEOFF_form.md).")
    print("     Do NOT fit. Wait for r128xp0.3 to reach >=4 seeds with >=10 vals, then re-run on the FULL 9-cell grid.")
    raise SystemExit

# Fit logit(acc)=a(r)-beta*(1-m)-beta1*(1-m)*log2(r), per-r intercepts
def logit(a): a=min(max(a,1e-4),1-1e-4); return math.log(a/(1-a))
rlist=sorted(set(x[0] for x in good)); ridx={r:i for i,r in enumerate(rlist)}
def design(x):
    r,p,m,acc,n,nv=x
    row=[0.0]*len(rlist); row[ridx[r]]=1.0            # a(r)
    row+=[-(1-m), -(1-m)*math.log2(r)]                # -beta*(1-m), -beta1*(1-m)*log2 r
    return row
X=np.array([design(x) for x in good]); y=np.array([logit(x[3]) for x in good])
coef,_,_,_=np.linalg.lstsq(X,y,rcond=None)
beta,beta1=coef[-2],coef[-1]
pred=X@coef; inmae=np.mean(np.abs(1/(1+np.exp(-pred)) - np.array([x[3] for x in good])))
# leave-one-cell-out
errs=[]
for i in range(len(good)):
    idx=[j for j in range(len(good)) if j!=i]
    c,_,_,_=np.linalg.lstsq(X[idx],y[idx],rcond=None)
    p_i=1/(1+math.exp(-(X[i]@c))); errs.append(abs(p_i-good[i][3]))
loco=float(np.mean(errs))
print(f"\n=== MATH-1.5B gated fit ({len(good)} cells, r={rlist}, n=5, >= {MINVALS} vals) ===")
print(f"  beta (noise slope)      = {beta:+.3f}")
print(f"  beta1 (compute x noise) = {beta1:+.3f}   ({'INTERACTION' if beta1<-0.05 else 'separable ~0'})")
print(f"  in-sample MAE = {inmae:.4f} | leave-one-cell-out MAE = {loco:.4f}")
print("  NOTE: firms the MATH interior point only; money plot stays PENDING (3B/APPS endpoints not converged).")
