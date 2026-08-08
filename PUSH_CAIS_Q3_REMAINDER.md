# CAIS job: Qwen-3B MATH remainder — finish the v4q3 master grid — 2026-08-08

**Why:** Della's Qwen-3B (`v4q3`) grid is bottlenecked at a 20-GPU/user QOS cap. An audit of all 240 cells
(recursive log scan, 2026-08-08) found **76 cells whose best run is still < step 234** — gaps, early
crashes, MIG fast-fails, and 48h-wall timeouts. CAIS (freed from the Falcon-3B grid) clears them fast.
Same W&B project (`rm4411-princeton-university/RLVR`) + **same run names** → Della-side analysis and
dedup-by-max-step slot them straight into the existing grid.

## The sweep
`sweeps/cais_v4q3_remainder.txt` — **76 cells**, format `RUN_NAME FP FN SEED ROLLOUTS MODELPATH`.
- Audit buckets (why each is here): **42 early-fail (<step 90), 17 mid (90–199), 12 near-complete/timeout
  (200–233), 5 MIG-gap (never trained)**.
- Rollouts: **32× r128, 16× r32, 28× r8**. Model = `MODELS/Qwen2.5-3B` (already staged on CAIS for the
  earlier fp0.45 fill).

## Pull — CHECKOUT-ONLY, do NOT `git pull`/merge
This branch is built on **Della's** tree and lacks CAIS infra — a merge would delete `jobscripts/grpo_cais.sh`,
`cais_env.sh`, `launch_sweep_cais.sh` and revert the `virtual_cluster.py` Ray-cgroup fix (`f8e3662`), breaking
the live campaign. Grab only the two new files:
```
git fetch origin
git checkout origin/backup/analysis-sweeps-infra-2026-07-11 -- \
  sweeps/cais_v4q3_remainder.txt PUSH_CAIS_Q3_REMAINDER.md
```

## Coordination — NO duplication
- The **other 148** v4q3 cells already have a clean 234 run — **do NOT re-run them**.
- Della is finishing **16** in-flight v4q3 cells (its 20-GPU slice) — those are **NOT** in this sweep.
- So: **CAIS 76 + Della 16 in-flight + 148 clean = 240 complete**, zero overlap.

## Launch (identical to how CAIS ran `cais_falcon_math` / `cais_q3_fp45_fill` — it's a MATH sweep)
Array over `grpo_cais.sh` with `PARAMS_FILE=sweeps/cais_v4q3_remainder.txt` and:
```
BASE_CONFIG=examples/configs/grpo_gsm8k_1B_rollout32_batch32.yaml   # gsm8k base cfg + MATH override
EXTRA_OVERRIDES="data.train.dataset_name=math data.validation.dataset_name=math \
  data.default.system_prompt_file=examples/prompts/math.txt \
  policy.train_micro_batch_size=1 policy.dtensor_cfg.activation_checkpointing=true \
  checkpointing.enabled=false"
```
- **WALLTIME: use CAIS's max = 48h** (CAIS QOS caps here — NOT Della's 60/72h). 48h is enough for a 3B r128
  on A100-80GB — the Falcon-3B r128 cells completed under this cap — so the redone timeout cells still reach a
  clean 234. Just confirm the r128 step-rate projects under 48h. r8/r32 finish in a few hours.
- CAIS is all A100-**80GB** → **no** `gpu80`/`nomig` constraint needed.
- Throttle the array `%` to available A100s. **Read PEAK accuracy for analysis** (MATH degrades late).
