# CAIS job: Falcon-3B MATH — finish the v4f3 r128 tail — 2026-08-13

**Why:** `v4f3` (Falcon-3B MATH) ran on CAIS. Audit of authoritative W&B `_step` (dedup max, 2026-08-13):
**r128 = 73/80 full-epoch (step 234); 7 SHORT and STALLED** (nothing running to finish them — they died on
CAIS and were never relaunched; Della can't complete them, wrong model path). The 7 are just **2 noise
conditions**: `fp0.0-fn0.15` (all 5 seeds) and `fp0.0-fn0.45` (s4, s5). Finishing them closes the Falcon-3B
r128 grid. Same W&B project + same `v4f3-*` names → dedup-by-max-step slots them in place.

## The sweep
`sweeps/cais_v4f3_r128_finish.txt` — **7 cells**, format `RUN_NAME FP FN SEED ROLLOUTS MODELPATH`.
Model = `MODELS/Falcon3-3B-Base` (already staged on CAIS). The other 73 r128 cells are done — do not re-run.

## Pull — CHECKOUT-ONLY, do NOT `git pull`/merge
Merging this Della-built branch would delete `jobscripts/grpo_cais.sh`, `cais_env.sh`, `launch_sweep_cais.sh`
and revert the Ray-cgroup fix (`f8e3662`). Grab only the two new files:
```
git fetch origin
git checkout origin/backup/analysis-sweeps-infra-2026-07-11 -- \
  sweeps/cais_v4f3_r128_finish.txt PUSH_CAIS_FALCON3B_R128.md
```

## Launch — identical to how the rest of v4f3 ran (from `cais_falcon_math`)
Array over `grpo_cais.sh` with `PARAMS_FILE=sweeps/cais_v4f3_r128_finish.txt` and:
```
BASE_CONFIG=examples/configs/grpo_gsm8k_1B_rollout32_batch32.yaml   # gsm8k base cfg + MATH override
EXTRA_OVERRIDES="data.train.dataset_name=math data.validation.dataset_name=math \
  data.default.system_prompt_file=examples/prompts/math.txt \
  policy.train_micro_batch_size=1 policy.dtensor_cfg.activation_checkpointing=true \
  checkpointing.enabled=false"
```
- NOISE_ENV defaults to `math` (noise via `++env.math.fp/fn`) — do not override.
- Full-epoch: the base config runs 1 epoch = **234 steps** (max_num_steps uncapped in the v4f3 config). Do
  NOT re-introduce a `max_num_steps=150` cap.
- Single A100-**80GB** per cell, no gpu80/nomig constraint. **WALLTIME 48h** — the 73 completed r128 cells
  prove Falcon-3B r128 reaches 234 within the CAIS cap.
- **Read PEAK accuracy** for analysis (MATH degrades late; Falcon plateaus ~step 100 but full 234 is needed
  for the completion standard + late-degradation).
