# CAIS + Della job split — 2026-07-25

**Division of labor (CAIS does NOT need to do everything):**
- **Della (Princeton):** runs **`v4f1` (Falcon-1B)** cells from `sweeps/cais_falcon_math.txt` — 1B fits any
  a100 (no gpu80), so it uses Della spare capacity without fighting the v4q3 3B priority. (Della already has
  the Falcon models + math dataset + launchers.)
- **CAIS (all a100-80GB):** runs **`v4f3` (Falcon-3B)** cells from the same sweep (needs 80GB) **+ the
  Qwen-3B fp0.45 mop-up** below.

## Second CAIS sweep: Qwen-3B fp0.45 fill (completes a CORE grid)
`sweeps/cais_q3_fp45_fill.txt` — **80 cells**: the fp/fn=0.45 margin cells of the v4q3 MATH master grid that
are missing or undertrained (<step90) — they got clipped by Della's 24h wall (esp. r128). Same `v4q3-*` run
names so they slot into the existing grid; MODEL=`MODELS/Qwen2.5-3B` (already staged on CAIS). **Give them
longer walltime so they train past the MATH peak (~step 90).** Higher-value than net-new cells — it finishes
a core Qwen family grid. Launch with the same MATH override as below.

---

# CAIS job: Falcon MATH master grid (third family) — 2026-07-25

**Why:** Falcon3 is a genuine non-Qwen/non-OLMo family that takes off on MATH under RLVR
(150-step single-seed probes on Della: **Falcon-3B 7%→66%**, **Falcon-1B 3%→17%**, and the
margin/collapse signal reproduced — Falcon-1B clean 16.8% vs noisy m=0.4 → 2.0%). Those were
n=1 probes; this promotes them to the **full master grid** so the collapse + β₁ get measured
in a third family. Runs land in the SAME W&B project (`rm4411-princeton-university/RLVR`), so
Della-side analysis pulls them automatically.

## The sweep
`sweeps/cais_falcon_math.txt` — **480 cells**, same format as `cais_v4q7_math.txt`
(`RUN_NAME FP FN SEED ROLLOUTS MODELPATH`):
- **`v4f1`** = Falcon3-1B-Base (240 cells) · **`v4f3`** = Falcon3-3B-Base (240 cells)
- master grid: fp/fn ∈ {0, 0.15, 0.3, 0.45}² × rollouts {8, 32, 128} × seeds 1–5

## Setup on CAIS (one-time)
1. **Models** → `MODELS/Falcon3-1B-Base` and `MODELS/Falcon3-3B-Base`
   (`hf download tiiuae/Falcon3-1B-Base` / `tiiuae/Falcon3-3B-Base` — open, ~3G/6G; both are
   `LlamaForCausalLM`, so vLLM/transformers load them with no arch work).
2. **Dataset** → the `math` key (EleutherAI/hendrycks_math) already staged for the Qwen-7B MATH runs.
3. CAIS is all A100-**80GB**, so **no gpu80/nomig constraint needed** (unlike Della, where Falcon-3B/Phi-2
   OOM'd on 40GB cards).

## Launch (identical to how `cais_v4q7_math` was launched — it's a MATH sweep)
Array over `grpo_cais.sh` with `PARAMS_FILE=sweeps/cais_falcon_math.txt` and the **MATH override**:
```
EXTRA_OVERRIDES="data.train.dataset_name=math data.validation.dataset_name=math \
  data.default.system_prompt_file=examples/prompts/math.txt \
  policy.train_micro_batch_size=1 policy.dtensor_cfg.activation_checkpointing=true \
  checkpointing.enabled=false"
BASE_CONFIG=examples/configs/grpo_gsm8k_1B_rollout32_batch32.yaml   # gsm8k base cfg + math override (NOT grpo_math_1B.yaml)
```
Single-GPU per cell; throttle the array `%` to available A100s. **Read PEAK accuracy** for analysis
(MATH degrades late — Falcon plateaued ~step 100).
