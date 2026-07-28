# RLVR noisy-verifier study — Claude handoff (2026-07-28)

Drop-in context for a fresh Claude/session (esp. if the account switches, since the auto-memory under
`~/.claude/.../memory/` does NOT transfer). Repo: `/scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards`,
branch `backup/analysis-sweeps-infra-2026-07-11` (this is the branch CAIS pulls). Cluster = Della
(login `della9`, internet on login node, compute nodes air-gapped). W&B project (all clusters):
`rm4411-princeton-university/RLVR`.

## 1. What the paper is
Empirical scaling-law characterization of GRPO-RLVR under a NOISY verifier (independent false-positive
`fp` / false-negative `fn` reward flips). Three findings:
- **(A) Margin collapse:** accuracy is organized by the margin **m = 1 − fp − fn** (Youden's J). `acc=f(m)`.
- **(B) Compute can't buy back supervision — the HEADLINE:** `logit(acc)=a0+a1·log2(r) − β(1−m) − β1(1−m)·log2(r)`
  with **β1 ≈ 0** ("rate in theory, fate in practice / you can't outspend a bad verifier"). The functional
  form IS the contribution; it lives/dies on HELD-OUT validation of β1≈0. **Held-out validation is NOT done
  yet — it's analysis on existing data, not a run. Highest-value open task.**
- **(C) FP/FN asymmetry:** base-rate-moderated; reported as a limitation (underpowered by design).
Positioned as the sober empirical counterweight to Rad ("Rate or Fate?", asymptotic theory) and Plesner
("An Imperfect Verifier is Good Enough", symmetric noise, single-seed) — we study *finite compute, ~1 epoch,
independent fp/fn, multi-seed*. NOT a methods contribution.

## 2. Grid design — see `GRID_DESIGN.md` (source of truth)
- **MASTER grid** = fp/fn ∈ {0, 0.15, 0.3, 0.45}² (16 cells) × rollouts {8,32,128} × seeds 1–5 = 240/model/task.
- **Supplements** (both on Qwen-1.5B + OLMo-1B): fine-margin `c1` (fp/fn {0…0.5}² @ r32) and compute-axis
  `ts05M` (fp/fn {0,0.15,0.3}² × rollouts {8,16,32,64,128}). Compute-axis is **diagonal-complete, off-diag
  optional — no rerun needed** (verified 2026-07-25).
- Base models only, NO Gemma-instruct in the grid. Task read from `config.data.default.system_prompt_file`
  (`gsm8k.txt`/`math.txt`) — the noise env is logged `env.math` for BOTH tasks; `dataset_name` is null.

## 3. What's RUNNING now + cluster ownership
- **Della:** `v4f1` = **Falcon-1B MATH grid** (240 cells, array `%10`, `--constraint=nomig`), ~10 running /
  ~165 pending. PLUS `v4q3` = **Qwen-3B MATH master grid** (~163/240 cells touched, ~110 trained-past-peak;
  the fp0.45 margin cells lag). v4q3 is currently STARVED (0 running / 143 pending) because Falcon-1B holds
  the full-a100 pool — a standing 1B-vs-3B tradeoff the user is aware of. Capacity fluctuates as other
  griffith users grab/release the shared a100 pool (we swing 0↔10 running).
- **CAIS = does ALL the 3Bs** (user's explicit rule; 7Bs are HELD). CAIS is A100-80GB, freed from 7Bs.
  Two sweeps PUSHED for CAIS to `git pull` + launch (see `PUSH_CAIS_FALCON.md`):
  - `sweeps/cais_falcon_math.txt` → CAIS runs the **`v4f3` (Falcon-3B)** half (240 cells). *(It also contains
    v4f1 lines; CAIS must launch ONLY v4f3 — Della owns v4f1. Footgun, not yet stripped.)*
  - `sweeps/cais_q3_fp45_fill.txt` → **Qwen-3B fp0.45 mop-up** (80 undertrained cells).
- **PLI (H100):** we have `v4q3_pli` backfill (`--account=pli_x --qos=pli-low --partition=pli --mem=180G`) but
  pli-low is backfill-starved — idle pli nodes are reserved for owners; we get ~nothing. NOT worth adding to.
- **Ionic:** separate CS cluster (ssh via `cycles`→`ionic`, Duo ControlMaster, bursty). Idle/unused; can host
  ≤1.5B only (a6000/a5000/3090), NOT 3B. `PORT_CAIS.md` + `PORT_ALLIANCE.md` cover cross-cluster ports.

## 4. Third-family exploration — this session's big result (single-seed 150-step PROBES, need gridding)
Searched for a 3rd base family (≤5B) beyond Qwen+OLMo. Final curves (base→final):
- **Falcon3-3B: MATH 7%→66%, CODE(KodCode-easy) 5.9%→73%** ✅ strong all-rounder.
- **Falcon3-1B: MATH 3%→17%, CODE 1.6%→30%** (modest/late-bloomer). Margin/collapse reproduced:
  Falcon-1B MATH clean 16.8% vs noisy(m=0.4) 2.0%.
- **Phi-1.5 CODE 1.6%→64%, Phi-2 CODE 3.9%→~40%** ✅ (Phi = code specialist). **Phi MATH weak** (Phi-1.5 flat
  ~3%, Phi-2 ~10%). **Phi-2 GSM8K base rate 23%** (strong, never RL-gridded).
- **Gemma-3 DROPPED** — base pt floors at 0% on MATH (tech-report 48 is the *instruct* model).
- **BIG LESSON (learned the hard way, repeatedly): weak base rate ≠ no takeoff, and never judge before
  ~step 50–80.** Phi-1.5 code 1.6%→64%; Falcon-1B code looked flat till step 70 then hit 30%. Do NOT
  pre-dismiss a run on its base rate.
- **CODE MODALITY IS RESURRECTED** (was "void/Qwen-only") — multiple non-Qwen families take off strongly.
  Next step = grid it (fp/fn × rollouts × seeds) on CAIS/Della to make it paper-citable.

## 5. Technical gotchas (will bite you)
- **MIG breaks vLLM:** vLLM 0.11.2 can't parse MIG device UUIDs (`ValueError: invalid literal for int():
  'MIG-...'`). ALL full-FT training needs **`--constraint=nomig`** (full a100). 1B does NOT get a free MIG lunch.
- **3B/2.7B+ need `--constraint=gpu80`** (nomig can hand a 40GB a100 → OOM at first train step; default vLLM
  util 0.6 doesn't fit 40GB). 1.3B fits 40GB fine.
- **MATH runs:** use BASE_CONFIG `examples/configs/grpo_gsm8k_1B_rollout32_batch32.yaml` (has proper
  train/validation + `_override_:true`) and override `data.train.dataset_name=math data.validation.dataset_name=math
  data.default.system_prompt_file=examples/prompts/math.txt`. Do NOT use `grpo_math_1B.yaml` (it uses
  OpenMathInstruct-2 = not cached offline + `validation:null`).
- **Datasets cached offline on Della:** `math`(EleutherAI/hendrycks_math), `gsm8k`(openai/gsm8k), and
  **KodCode** (had to `load_dataset('KodCode/KodCode-Light-RL-10K')` on the login node into `$SCR/huggingface`).
- **Code (KodCode):** env_name=`code_verify`, needs `/usr/bin/bwrap` (present on Della, unknown on CAIS —
  CHECK before code-gridding on CAIS). Grader is `nemo_rl/environments/code_verify.py::run_kodcode`; loader
  `nemo_rl/data/datasets/response_datasets/kodcode.py` (KODCODE_DIFFICULTY env). `+env.code_verify.num_workers=8`,
  `gpu_memory_utilization=0.3`, `max_model_len` ≤ model context (Phi=2048).
- **MATH degrades late** (peaks ~step 90 then drifts down) — for analysis read PEAK accuracy, not final.
- **W&B:** match cells by run NAME (identical across clusters), NOT `config.policy.model_name` (path differs
  per cluster). Bulk config scans time out; use targeted server-side count queries. r128 caps ~190–213 steps
  (don't require `_step>=225`).

## 6. Working launchers / files
- `jobscripts/grpo_della.sh` — the workhorse (reads PARAMS_FILE array + BASE_CONFIG + EXTRA_OVERRIDES + NOISE_ENV
  from env; MODEL from sweep col 6). PLI submit adds `--account=pli_x --qos=pli-low --partition=pli --mem=180G`.
- Session launchers: `jobscripts/falcon_rl_probe.sh`, `falcon_code_rl.sh`, `smoke_newfam.sh`.
- Sweeps: `sweeps/della_falcon1b_math.txt` (running), `sweeps/cais_falcon_math.txt`, `sweeps/cais_q3_fp45_fill.txt`.
- Docs: `GRID_DESIGN.md`, `PUSH_CAIS_FALCON.md`, `PORT_CAIS.md`, `PORT_ALLIANCE.md`.
- Uncommitted code changes in the working tree: `nemo_rl/environments/code_verify.py` (+run_kodcode),
  `nemo_rl/data/datasets/response_datasets/{__init__,kodcode,openr1code}.py`, `test_sandbox_containment.py`.
  These are needed for CODE grids (not MATH). Not yet committed.

## 7. Open tasks (priority order)
1. **Held-out validation of β1≈0** on the completed Qwen/OLMo MATH+GSM8K grids (analysis, no runs). Load-bearing.
2. **CAIS: launch the two pushed sweeps** (Falcon-3B `v4f3` + Qwen-3B fp0.45 fill).
3. **Finish Falcon-1B grid** on Della (running) + the Qwen-3B MATH grid (esp. the lagging fp0.45 cells).
4. **Grid the code modality** (Falcon-3B + Phi-1.5 on KodCode, fp/fn × rollouts × seeds) — resurrected & strong.
5. Optionally: Phi-2 GSM8K grid (23% base, never RL'd); Phi/Falcon supplements.

## 8. Operating norms the user cares about (learned this session)
- **Don't thrash.** Hold the agreed plan steady; treat a question as a question, not a trigger to launch things.
- **CAIS does all 3B; Della does 1B.** Don't put 3B on Princeton or duplicate an assignment across clusters.
- **Keep the 3Bs at the front of the line** (priority), but the user explicitly wanted Falcon-1B running on Della.
- Cron: 90-min status checks (leg A/B) — make NO config/walltime/step changes, do NOT scancel, just report.
- **Halt-all safeguard:** do NOT `scancel` all jobs until asked 10 separate times (EMNLP). Fixing/relaunching a
  specific broken cell of your own axis IS allowed when authorized.
- Never print the W&B API key. Untrusted code MUST run in the bwrap sandbox (fail-closed).
