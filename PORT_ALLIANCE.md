# Porting the noisy-RLVR pipeline to a Digital Research Alliance of Canada H100 cluster

**Why:** access via a Canadian PI (family) → CCDB sponsorship. H100 fleet dwarfs Della (A100)/neuronic (L40).
**Target order:** Nibi (internet compute nodes = easiest) → Rorqual / Trillium (air-gapped, but we already solved that).
**Skip:** Narval (A100-40GB, 2021 — lateral to Della, no upgrade).

## 0. Access (dad does this, ~1h to enable)
- Dad (PI) logs into CCDB, adds you to his group / you request access under his allocation account (`def-<pi>`).
- You: CCDB → Resources → Access Systems → request **Nibi, Rorqual, Trillium** (all three; fair-share, use whichever's free).
- Set up SSH keys + MFA (all Alliance clusters require it; Trillium disables passwords entirely).

## 1. Cluster-specific facts that change our scripts
| | Nibi | Rorqual | Trillium |
|---|---|---|---|
| GPU | 288× H100-80GB, **8/node** | 372× H100-80GB, 4/node | 252× H100-80GB, 4/node |
| compute-node internet | **YES** (easy port) | no (air-gap) | no (air-gap) |
| scratch | 1TB soft (60d grace) — TIGHT | large, purged | 25TB, no purge yet |
| crontab | — | **none** | — |
| caps | — | ≤1000 jobs, ≤7d wall | scheduled by whole GPU/node |

## 2. Data to stage (our footprint ~600G)
- **Models** (`hf_models/`): OLMo-2-0425-1B, OLMo-2-1124-7B, Qwen2.5-0.5B/1.5B/3B/7B. On **Nibi**: download direct from HF (compute has internet). On Rorqual/Trillium: transfer via **Globus** (endpoints `alliancecan#nibi` etc.) or download on login node.
- **Datasets**: GSM8K, MATH, CodeContests — same (direct on Nibi, Globus/login elsewhere).
- Put models + data in **PROJECT** (backed up, per-group quota) NOT scratch on Nibi (1TB scratch too tight for models).

## 3. The hard part — rebuild the NeMo-RL venv
- Della's env is a prebuilt uv venv (air-gapped, worker venvs). Alliance uses `StdEnv/2023` + module system (`module load python cuda ...`).
- Rebuild the nemo-rl venv against Alliance modules. **On Nibi (internet) this is far easier** (pip can fetch). On air-gapped ones, mirror the Della offline-wheel approach.
- H100 = Hopper: our Della build already fought "Hopper-only pkgs" (flash-attn etc.) — on native H100 those should resolve *cleaner*, not harder.
- vLLM + Ray + FSDP/DTensor all CUDA — fine on H100.

## 4. Jobscript changes (grpo_della.sh → grpo_alliance.sh)
- GPU request: `--gres=gpu:N` → **`--gpus-per-node=h100:4`** (or `--gpus=h100:N` for "anywhere").
- Account: add **`--account=def-<pi>`**.
- No Della-style QOS/partition routing; Alliance = account + gpu spec.
- **HOME/PROJECT not writable from compute jobs** → `logger.log_dir` and checkpoint dirs must point to **SCRATCH**. Fix the log paths.
- W&B: Nibi = **online** live logging works; Rorqual/Trillium = offline + `sync_wandb.sh` (as on Della).
- Walltime ≤7d, ≤1000 queued jobs — fine (our arrays ≤240, jobs ≤24h).
- Whole-GPU scheduling, no MIG to dodge for full GPUs.

## 5. First smoke on arrival
- 1 clean cell (fp=0/fn=0, r=8) on a single H100 to validate the venv + noise env + W&B, THEN the 4-GPU 7B smoke, THEN launch grids.

## What this unlocks
H100 ≈ 2–3× A100; hundreds of them. The entire remaining campaign (all MATH grids + OLMo-7B + Qwen-7B) would finish in a
fraction of Della's time, and the 4-GPU jobs become trivial (8-GPU Nibi nodes / abundant H100). Della stays as-is; Alliance
is pure additional throughput.
