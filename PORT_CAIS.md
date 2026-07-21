# Porting the noisy-RLVR pipeline to the CAIS cluster (A100-80GB) — for the on-cluster agent

**Why CAIS first:** already have access, ~40 A100-80GB free right now (incl. a fully-idle 8-GPU node),
Slurm, and it's A100 like Della so the port is nearly copy-paste (vs the H100 rebuild the Alliance clusters need).
**Partition:** `cais`. Confirmed launchable now: `salloc --partition=cais --gres=gpu:N`.

## 0. Get the code (trivial — it's on GitHub)
```
git clone <the rm-3284/RLVR_noisy_rewards repo, branch backup/analysis-sweeps-infra-2026-07-11>
```
Everything is there: jobscripts/, sweeps/ (params), examples/configs/, analysis/, della_env.sh.

## 1. Stage data (the one real time cost)
- **Models** (~356G): OLMo-2-0425-1B, OLMo-2-1124-7B, Qwen2.5-0.5B/1.5B/3B/7B.
  - If CAIS compute/login has **internet** → `huggingface-cli download` each straight to WekaFS. Easiest.
  - If air-gapped → rsync from Della `/scratch/gpfs/GRIFFITHS/aw2418/hf_models/` (or Globus).
- **Datasets**: GSM8K, MATH (CodeContests later). Same — HF download or rsync.
- Put models+data on **WekaFS** (shared) so all nodes see them. Note the CAIS path; it replaces the Della
  `/scratch/gpfs/GRIFFITHS/aw2418/hf_models/...` prefix used in the params files (see step 3).

## 2. Rebuild the venv (A100/Ubuntu 22.04 — like Della, straightforward)
- Recreate the nemo-rl uv venv (see della_env.sh for the env vars: UV_CACHE_DIR, HF_HOME, PYTHONPATH, etc.).
- A100 = Ampere (not Hopper), so NONE of the Della "Hopper-only pkg" fights apply — should build clean.
- vLLM + Ray + FSDP/DTensor all standard on A100.

## 3. Jobscript changes (grpo_della.sh → grpo_cais.sh)
grpo_della.sh already reads PARAMS_FILE / BASE_CONFIG / EXTRA_OVERRIDES from env and the MODEL PATH from
params-file column 6. So the only changes:
- SBATCH: `--partition=cais` (drop Della's QOS routing), `--gres=gpu:N`, appropriate `--mem`, `--cpus-per-task`.
- Model paths: the params files use absolute Della paths in col 6. Either (a) rsync models to the SAME path on
  CAIS, or (b) sed the params to the CAIS WekaFS model dir. Option (a) is zero-edit.
- Logs: point `logger.log_dir` / checkpoint dirs to CAIS scratch (WekaFS or local NVMe /27TB per node).
- W&B: if CAIS has internet → online logging works (project `rm4411-princeton-university/RLVR`, same as Della,
  so runs land in the SAME W&B and Claude-on-Della can pull/analyze them). If air-gapped → offline + sync_wandb.sh.

## 4. First launches (priority order — put the STUCK stuff here)
1. **7B grids on the idle 8-GPU node** — OLMo-7B (`v4o7`) + Qwen-7B/GSM8K (`q7bgsm`), 4-GPU each, 2 per node.
   Use the sub_ailab_fsdp override set: `cluster.gpus_per_node=4 policy.dtensor_cfg.tensor_parallel_size=1
   policy.generation.vllm_cfg.tensor_parallel_size=4 policy.train_micro_batch_size=1
   policy.dtensor_cfg.activation_checkpointing=true`, mem ~360G.
2. **Qwen MATH grids** (v4q05/v4q15/v4q3pli) single-GPU on the scattered free A100s — the MATH-collapse crux.
3. **Thin-cell re-runs** (the r=128 MATH cells that undertrained on Della's 24h wall) — give them longer walltime
   here so they train past the wall. NOTE: MATH peaks ~step 90 then DEGRADES (late-degradation), so for analysis
   read PEAK accuracy, not final — don't just train longer blindly.

## Coordination
- CAIS-side agent: staging + venv + launches + on-cluster monitoring (Della-anchored Claude is firewalled from CAIS).
- Della-side Claude: keeps Della running, prepped these artifacts, and can pull/analyze ALL results via W&B
  (reachable from anywhere) — so cross-cluster analysis still works from one place.
- Runs use the SAME W&B project → one unified dataset regardless of which cluster produced a cell.
