# CAIS job: OLMo-1B MATH — finish the v4o1 r128 tail — 2026-08-13

**Why:** OLMo-1B is the study's lowest-base-rate anchor (MATH base ~1%). The master OLMo MATH grid is
`v4o1` (full 16-cell fp/fn design × 3 rollouts × 5 seeds, already on CAIS). Audit of authoritative W&B
`_step` (dedup max, 2026-08-13):

- **r8: 80/80 cells full-epoch (step 234) — DONE, do not touch.**
- **r32: 78/80 full-epoch — DONE except 2 cells.**
- **r128: 34/80 full-epoch, 46 SHORT (mostly step 120–160/234).**

The reduced Della `olmomath-*` grid (r32 capped at 109, r128 died) is **superseded by v4o1** — ignore it.
This job finishes **only the short tail** so the whole v4o1 grid hits full-epoch. Same W&B project
(`rm4411-princeton-university/RLVR`) + **same run names** → dedup-by-max-step slots them in place.

## The sweep
`sweeps/cais_v4o1_r128_redo.txt` — **48 cells** (46× r128 + 2× r32), format `RUN_NAME FP FN SEED ROLLOUTS MODELPATH`.
- Every v4o1 r32/r128 cell with max W&B `_step` < 232. Model = `MODELS/OLMo-2-0425-1B` (already staged on CAIS,
  same path as the `cais_v4o1_missing` sweep).
- The 158 already-full cells (all r8, 78 r32, 34 r128) are **left out** — do not re-run them.

## Pull — CHECKOUT-ONLY, do NOT `git pull`/merge
Merging this Della-built branch would delete `jobscripts/grpo_cais.sh`, `cais_env.sh`, `launch_sweep_cais.sh`
and revert the Ray-cgroup fix (`f8e3662`). Grab only the two new files:
```
git fetch origin
git checkout origin/backup/analysis-sweeps-infra-2026-07-11 -- \
  sweeps/cais_v4o1_r128_redo.txt PUSH_CAIS_OLMO_MATH.md
```

## Launch (identical config to how CAIS ran the rest of v4o1 — it's a MATH sweep)
Array over `grpo_cais.sh` with `PARAMS_FILE=sweeps/cais_v4o1_r128_redo.txt` and:
```
BASE_CONFIG=examples/configs/grpo_gsm8k_1B_rollout32_batch32.yaml   # gsm8k base cfg + MATH override
EXTRA_OVERRIDES="data.train.dataset_name=math data.validation.dataset_name=math \
  data.default.system_prompt_file=examples/prompts/math.txt \
  policy.train_micro_batch_size=1 policy.dtensor_cfg.activation_checkpointing=true \
  checkpointing.enabled=false"
```
- **NOISE_ENV defaults to `math`** (grpo_cais mirrors grpo_della) — noise applied via `++env.math.fp/fn`. Do
  not override it.
- **WALLTIME: CAIS max = 48h.** OLMo-1B is tiny; the 34 v4o1 r128 cells that already reached 234 prove it
  completes on CAIS A100-80GB well under this cap. r32 finishes in a couple hours.
- CAIS is all A100-**80GB** → no `gpu80`/`nomig` constraint. Throttle the array `%` to available A100s.
- **Read PEAK accuracy for analysis** (OLMo MATH tops out ~7–9% clean; noise collapses it toward ~3%).
