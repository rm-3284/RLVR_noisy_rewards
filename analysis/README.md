# analysis/

Post-hoc fits + diagnostics for the noisy-RLVR study. Each script is **standalone** (pulls its
own runs from W&B `rm4411-princeton-university/RLVR`). Run from repo root:
`source della_env.sh && python analysis/<sub>/<script>.py`.

**Data source of truth:** [`../INVENTORY.md`](../INVENTORY.md) — every usable grid
(model × task × rollout × cells). Regenerate from `logs/` any time.

Reorganized into purpose subdirs 2026-07-20 (scripts were flat before; paths in older logs prose
may still cite the flat location).

## collapse/ — does acc depend only on margin m = 1 − fp − fn?
- `matched_margin_all_configs.py` — matched-margin test across **every** off-diagonal config (survey).
- `fit_forms_clean.py` — **canonical** collapse / FP-FN asymmetry (M1/M2/M3), leave-one-config-out, marginalizes over r.
- `fit_collapse_olmo_gsm8k.py` — OLMo-1B/GSM8K collapse @ r=8, seed-level + cluster-bootstrap asymmetry CI.
- `fit_collapse_olmo_gsm8k_allr.py` — same **per rollout** (r8/32/128). Shows collapse is **strong-but-not-strict**:
  spread ~constant (~0.05) but the test SHARPENS with compute (seed noise shrinks), so strict test passes r8, fails r32/r128.

## law/ — closed-form tradeoff  logit(acc) = a0 + a1·log2(r) − β(1−m) − β1(1−m)·log2(r)
- `fit_law_closedform.py` — **β1 fitter OF RECORD. CHEATING-FREE**: one parametric a1·log2r compute term, **NO free
  per-r intercepts** (free intercepts inflate β1 — that was the archived lineage's bug). Held-out on LOCO (leave-cell)
  AND LORO (leave-r / extrapolate). Has per-config entries.
- `fit_beta1_olmo_gsm8k.py` — β1 on the full OLMo-1B/GSM8K cube (also cheating-free parametric form). Change the name
  regex to run it on any multi-rollout grid (used for the Qwen-3B/GSM8K cross-family point).
- `fit_ceiling_saturation.py` — ceiling/floor-saturation boundary fit.

## diagnostics/
- `power_check_asymmetry.py` — plant-known-γ power sim; a 16-cell grid can only resolve |γ| ≳ 0.5.

## util/
- `extract_metrics.py` (runs→CSV), `make_figures.py` (paper figs), `curve_fitting_v4.py` (compute curves incl. Gemma).

## archive/
Superseded probes + the **free-intercept β1 lineage** (`fit_math_beta1.py` → `fit_math_settled.py` →
`fit_math_crossfam.py`) whose β1 numbers are inflated by the nonparametric a(r). Provenance only — **do not build on these.**

---

## β1 result of record (cheating-free closed form)

| config | base | β1 | held-out interaction? | date |
|---|---|---|---|---|
| MATH-0.5B  | 0.31 | −0.263 | ✅ (LOCO+LORO) | 07-10 |
| GSM8K-0.5B | 0.48 | −0.256 | ❌ null ties (grid thin) | 07-10 |
| MATH-1.5B  | 0.62 | −0.079 | ❌ separable | 07-10 |
| GSM8K-1.5B | 0.78 | +0.061 | ❌ separable | 07-10 |
| **GSM8K-OLMo-1B** | 0.54 | **−0.04** (CI [−0.15,+0.03]) | ❌ separable (full cube, 2nd family) | **07-20** |
| **GSM8K-Qwen-3B**  | — | **−0.03** (CI [−0.14,+0.10]) | ❌ separable | **07-20** |

**Reading:** the compute×noise interaction (β1<0) holds out on **MATH-0.5B only** → the ≥2-config replication bar is
**NOT yet cleared**. Everything on **GSM8K is separable (β1≈0)** — now confirmed across 4 configs and both families.
Direction of β1(base) is right (interaction at low base, separable at high) but not held-out-replicated.

## Next fit (the open shot)
**OLMo-1B/MATH** (base ~0.05, the lowest) is the next candidate for a 2nd held-out low-base interaction config — the full
16-cell × r{8,32,128} cube now exists (`v4o1-*` / `v4o1ail-*`; see INVENTORY.md, MATH/OLMo-1B). Its diagonal `m=1−2fp`
must be generalized to `m=1−fp−fn` before wiring into `fit_law_closedform.py`. Ties [[replication-adequacy]],
[[functional-forms-rigor]], [[key-finding-base-rate]].
