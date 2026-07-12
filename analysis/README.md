# analysis/

All post-hoc analysis of the RLVR noisy-verifier sweeps. **Run every script from the repo
root** (`python analysis/<script>.py`) — they glob `logs/...` with paths relative to CWD.

## Canonical scripts (use these)

| script | what it fits | form |
|---|---|---|
| `fit_law_closedform.py` | **β1 compute×noise coupling, CHEATING-FREE.** One explicit closed form `logit(acc)=a0+a1·log2r−β(1−m)−β1(1−m)log2r` — 4 coefficients, **no free per-r intercepts**. Held-out on LOCO (leave-cell) AND LORO (leave-r / extrapolate compute) vs separable null. Covers MATH + GSM8K (0.5B/1.5B). **This is the β1 fitter of record.** | tradeoff (B) |
| `fit_forms_clean.py` | **margin collapse / FP-FN asymmetry** (M1/M2/M3), leave-one-config-out. Marginalizes over r. | collapse (A) |
| `curve_fitting_v4.py` | clean compute curve fits (incl. Gemma). | — |
| `extract_metrics.py` | pull fp/fn/acc per run → CSV. | — |
| `make_figures.py` | render paper figures into `notebook/figures/`. | — |

## archive/ — superseded generations & one-off probes (kept for provenance, do not use)

- **β1 lineage (all FREE-INTERCEPT = the "cheat" a(r), superseded by `fit_law_closedform.py`):**
  `fit_math_beta1.py` → `fit_math_settled.py` → `fit_math_crossfam.py`. Their β1 numbers (e.g. GSM8K-0.5B
  −0.44) are inflated by the nonparametric a(r); see the 2026-07-10 correction block in `logs/TRADEOFF_form.md`.
- **collapse lineage:** `analyze_c1.py` → `fit_c1.py` → `fit_forms.py` → *(fit_forms_clean.py, canonical)*
- **compute/saturation:** `fit_c2.py`, `fit_c2_fast.py`
- **curve fitting:** `curve_fitting_v3.py` → *(curve_fitting_v4.py, canonical)*
- **one-off probes:** `probe_3b.py`, `probe_olmo.py`, `inspect_gemma_fmt.py`, `instruct_probe.py`

> Log prose (`logs/STRATEGY.md`, `logs/TRADEOFF_form.md`, detector files) still cites old script paths
> like `notebook/fit_math_crossfam.py` — those refer to the now-archived free-intercept lineage.

## Result of record (2026-07-10, cheating-free closed form)

| config | base | β1 | held-out |
|---|---|---|---|
| MATH-0.5B | 0.31 | −0.263 | ✅ interaction (LOCO+LORO) |
| GSM8K-0.5B | 0.48 | −0.256 | ❌ null ties (grid thin) |
| MATH-1.5B | 0.62 | −0.079 | ❌ separable |
| GSM8K-1.5B | 0.78 | +0.061 | ❌ separable |

Interaction held-out on **MATH-0.5B only** → ≥2-config bar NOT cleared. Direction of β1(base) right,
not held-out-replicated. See [[closed-form-no-cheating]], [[functional-forms-rigor]], [[replication-adequacy]].

## Known TODO (in `fit_law_closedform.py`)

Add the OLMo cross-family points. OLMo-1B lives in `v4o1-*` (NOT the dead `olmomath-*`; its r128 crashed).
v4o1 is the fp=0 asymmetric column, so the design's diagonal `m=1−2fp` must be generalized to `m=1−fp−fn`
before wiring it in. OLMo-1B (base ~0.05) is the next shot at a 2nd held-out low-base interaction config.
