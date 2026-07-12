"""Preliminary C1 gate analysis from W&B. Reports directional reads, seed spread (power),
margin collapse, r8-vs-r32 asymmetry, and red flags (broken cells, under-convergence, reward hacking).
Run anytime; it uses whatever runs have finished so far. Usage: python jobscripts/analyze_c1.py
"""
import os, statistics as st
from collections import defaultdict
import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "rm4411-princeton-university")
api = wandb.Api()
runs = api.runs(f"{ENTITY}/RLVR")

# bucket -> (fp,fn) -> list of dicts
data = {"r32": defaultdict(list), "r8": defaultdict(list)}
for r in runs:
    if r.name.startswith("c1r8-1.5B"): b = "r8"
    elif r.name.startswith("c1-1.5B"): b = "r32"
    else: continue
    em = r.config.get("env", {}).get("math", {})
    fp, fn = em.get("fp"), em.get("fn")
    acc = r.summary.get("validation/accuracy")
    tr = r.summary.get("train/reward")
    if fp is None or acc is None: continue
    # best val across training (under-convergence check); history() is one call vs row-by-row scan
    best = None
    try:
        h = r.history(keys=["validation/accuracy"], samples=300, pandas=False)
        vals = [row.get("validation/accuracy") for row in h if row.get("validation/accuracy") is not None]
        best = max(vals) if vals else acc
    except Exception:
        best = acc
    data[b][(round(float(fp),2), round(float(fn),2))].append(
        dict(acc=float(acc), best=(best if best is not None else float(acc)),
             tr=(float(tr) if tr is not None else None), state=r.state))

def cell_stats(lst):
    a = [x["acc"] for x in lst]
    return len(a), st.mean(a), (st.pstdev(a) if len(a) > 1 else 0.0)

for b in ("r32", "r8"):
    d = data[b]
    if not d: continue
    print(f"\n{'='*64}\n{b.upper()}  ({sum(len(v) for v in d.values())} finished runs, {len(d)} cells)\n{'='*64}")

    # coverage
    print("coverage (seeds per cell):")
    for (fp,fn) in sorted(d):
        n,m,s = cell_stats(d[(fp,fn)])
        print(f"  fp={fp} fn={fn}: n={n}  acc={m:.3f}±{s:.3f}")

    # FN marginal (fp=0) and FP marginal (fn=0)
    for axis,fixed in (("FN marginal (fp=0)", lambda k: k[0]==0.0), ("FP marginal (fn=0)", lambda k: k[1]==0.0)):
        cells = sorted([k for k in d if fixed(k)], key=lambda k: k[0]+k[1])
        if len(cells) >= 2:
            print(f"\n{axis}:")
            for k in cells:
                n,m,s = cell_stats(d[k]); rate = k[1] if "FN" in axis else k[0]
                print(f"  rate={rate}: acc={m:.3f}±{s:.3f} (n={n})")

    # asymmetry anchors: (a,0) vs (0,a)
    print("\nasymmetry anchors (FP vs FN at matched rate):")
    for a in (0.1,0.2,0.3,0.4,0.5):
        fpk,fnk = (a,0.0),(0.0,a)
        if fpk in d and fnk in d:
            nf,mf,sf = cell_stats(d[fpk]); nn,mn,sn = cell_stats(d[fnk])
            diff = mf-mn; pooled = (sf+sn)/2 + 1e-9
            tag = "FP better" if diff>0 else "FN better"
            sep = "SEPARATED" if abs(diff) > 2*pooled else "within noise"
            print(f"  rate={a}: FP(a,0)={mf:.3f}  FN(0,a)={mn:.3f}  diff={diff:+.3f} ({tag}, {sep})")

    # margin collapse: same m=1-fp-fn, different (fp,fn)
    bym = defaultdict(list)
    for (fp,fn) in d:
        bym[round(1-fp-fn,2)].append((fp,fn))
    collapse = [(m,cs) for m,cs in bym.items() if len(cs)>=2]
    if collapse:
        print("\nmargin-collapse check (same m, different cells):")
        for m,cs in sorted(collapse, reverse=True):
            accs = [(c, cell_stats(d[c])[1]) for c in cs]
            spread = max(a for _,a in accs)-min(a for _,a in accs)
            print(f"  m={m}: " + "  ".join(f"{c}={a:.3f}" for c,a in accs) + f"   spread={spread:.3f}")

    # red flags
    flags=[]
    for k in d:
        n,m,s = cell_stats(d[k])
        if m < 0.40: flags.append(f"LOW acc {m:.3f} at {k} (possible collapse/bug)")
        if n>1 and s > 0.05: flags.append(f"WIDE seed spread {s:.3f} at {k} (power: may need more seeds)")
        gap = st.mean([x['best']-x['acc'] for x in d[k]])
        if gap > 0.03: flags.append(f"UNDER-CONVERGENCE at {k}: best-final gap {gap:.3f}")
        trs=[x['tr'] for x in d[k] if x['tr'] is not None]
        if trs and (k[0]>0):  # high fp: noisy train reward >> true val acc = reward hacking
            if st.mean(trs)-m > 0.10: flags.append(f"REWARD-HACK signature at {k}: train_reward {st.mean(trs):.3f} >> val {m:.3f}")
    if flags:
        print("\nFLAGS:")
        for f in sorted(set(flags)): print("  - "+f)
    else:
        print("\nFLAGS: none")
