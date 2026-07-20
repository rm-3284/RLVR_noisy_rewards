"""Run the SAME matched-margin collapse test across every config that has OFF-DIAGONAL cells.

The margin-collapse claim is only testable where fp != fn: on the diagonal, d = fp-fn is 0
everywhere, so 'acc depends only on m' is unfalsifiable. This script finds every config with
>=1 matched-margin GROUP (two or more cells sharing m = 1-fp-fn) and reports, per config:

  * how many matched-margin groups exist (0 => that config CANNOT test collapse)
  * worst spread within a group, vs the within-cell seed-noise floor
  * PASS (spread <= 2*seed SD) / FAIL

Purpose: establish whether the collapse replicates on >=2 INDEPENDENT configs, per the
standing replication bar (seeds are not configs).
"""
import statistics as st
from collections import defaultdict
import numpy as np
import wandb

api = wandb.Api()
PROJECT = "rm4411-princeton-university/RLVR"

CONFIGS = {
    # the new, purpose-built 4x4 off-diagonal grid (OLMo-2-1B / GSM8K)
    "OLMo1B-GSM8K r=8":  r"^aiOLMOgsm-1B-r8-fp",
    "OLMo1B-GSM8K r=32": r"^aiOLMOgsm-1B-r32-fp",
    # the pre-existing configs from fit_forms_clean.py (9 converged)
    "gsm8k-1.5B":     r"^c1-1.5B-fp",
    "gsm8k-0.5B":     r"^p05B-gsm8k-fp",
    "gsm8k-3B":       r"^p3B-gsm8k-fp",
    "math-0.5B":      r"^pMATH-0.5B-fp",
    "olmo-gsm8k":     r"^olmo-gsm8k-fp",
    "olmo-math":      r"^olmomath-fp",
    "ext-1.5B":       r"^ext-gsm8k-1.5B-fp",
    "math-1.5B-mL4":  r"^mL4-1.5B-fp",
    "math-1.5B-mL5":  r"^mL5-1.5B-fp",
    # the in-flight v4 grids (Qwen, MATH) — partial, included to see how far along they are
    "v4 Qwen-0.5B MATH": r"^v4q05-",
    "v4 Qwen-1.5B MATH": r"^v4q15-",
    "v4 Qwen-3B MATH":   r"^v4q3pli-",
    "v4 OLMo-1B MATH":   r"^v4o1-",
}


def pull(rgx):
    best = {}
    for r in api.runs(PROJECT, filters={"display_name": {"$regex": rgx}}, per_page=500):
        em = r.config.get("env", {}).get("math", {})
        fp, fn = em.get("fp"), em.get("fn")
        acc = r.summary.get("validation/accuracy")
        if fp is None or fn is None or acc is None:
            continue
        step = r.summary.get("_step", 0) or 0
        if r.name not in best or step > best[r.name][0]:
            best[r.name] = (step, round(float(fp), 2), round(float(fn), 2), float(acc))
    if not best:
        return {}
    maxstep = max(v[0] for v in best.values())
    cells = defaultdict(list)
    for step, fp, fn, acc in best.values():
        if step < 0.6 * maxstep:
            continue
        cells[(fp, fn)].append(acc)
    return cells


print(f"{'config':<22} {'cells':>5} {'offdiag':>7} {'groups':>6} {'seedSD':>7} "
      f"{'worst':>7} {'2xSD':>7}  verdict")
print("-" * 88)
for name, rgx in CONFIGS.items():
    cells = pull(rgx)
    if not cells:
        print(f"{name:<22} {'-':>5} {'-':>7} {'-':>6} {'-':>7} {'-':>7} {'-':>7}  NO DATA")
        continue
    offdiag = sum(1 for (fp, fn) in cells if fp != fn)
    sds = [st.stdev(v) for v in cells.values() if len(v) > 1]
    seed_sd = float(np.mean(sds)) if sds else float("nan")

    bym = defaultdict(list)
    for (fp, fn), accs in cells.items():
        bym[round(1 - fp - fn, 2)].append(st.mean(accs))
    groups = {m: v for m, v in bym.items() if len(v) >= 2}
    if not groups:
        print(f"{name:<22} {len(cells):>5} {offdiag:>7} {0:>6} {seed_sd:>7.4f} "
              f"{'-':>7} {'-':>7}  CANNOT TEST (no matched margins)")
        continue
    worst = max(max(v) - min(v) for v in groups.values())
    ok = worst <= 2 * seed_sd
    print(f"{name:<22} {len(cells):>5} {offdiag:>7} {len(groups):>6} {seed_sd:>7.4f} "
          f"{worst:>7.4f} {2*seed_sd:>7.4f}  {'PASS' if ok else 'FAIL'}")

print("""
'groups' = number of margins m at which TWO OR MORE different (fp,fn) cells exist.
groups=0 means that config is diagonal-only and CANNOT test the collapse at all — it can
neither support nor refute it. Only configs with groups>=1 count toward replication.""")
