#!/bin/bash -l
#SBATCH --job-name=rlvr
#SBATCH --account=griffith
# NOTE: do NOT set --partition on this cluster — the submit plugin rejects explicit gpu/gputest
# and auto-routes by QOS (gpu-test->gputest, gpu-short->gpu). Pass --qos / --constraint at submit time.
# QOS is auto-assigned by --time per Della docs (gpu-short<=24h, gpu-medium<=72h, gpu-long<=144h) — do NOT hardcode --qos.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G   # refit/offload after each step peaks CPU RAM ~50G for 1.5B; headroom for larger models
#SBATCH --time=16:00:00
#SBATCH --mail-type=fail
#SBATCH --mail-user=aw2418@princeton.edu
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#
# Parameterized GRPO run on Della. Submit via jobscripts/launch_sweep.sh, which exports:
#   FP, FN, SEED, ROLLOUTS, MODEL, RUN_NAME   (and optionally BASE_CONFIG)
# Defaults below let it also run standalone for a smoke test.

set -euo pipefail
source /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards/della_env.sh

# ---- array mode: read this job's params from a sweep file ----
# Each line: RUN_NAME FP FN SEED ROLLOUTS MODEL   (line N+1 for array task N)
if [[ -n "${PARAMS_FILE:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS_FILE")
  read -r RUN_NAME FP FN SEED ROLLOUTS MODEL <<< "$line"
fi

# ---- parameters (env-var overridable) ----
FP="${FP:-0.0}"
FN="${FN:-0.0}"
SEED="${SEED:-42}"
ROLLOUTS="${ROLLOUTS:-32}"
MODEL="${MODEL:-/scratch/gpfs/GRIFFITHS/aw2418/hf_models/Qwen2.5-1.5B}"
RUN_NAME="${RUN_NAME:-rlvr-smoke}"
BASE_CONFIG="${BASE_CONFIG:-examples/configs/grpo_gsm8k_1B_rollout32_batch32.yaml}"
NUM_PROMPTS=32                                   # batch fixed at 32 (spec §1.2)
GLOBAL_BATCH=$(( NUM_PROMPTS * ROLLOUTS ))        # num_prompts_per_step * num_generations_per_prompt

# ---- air-gapped compute node: everything offline ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_ENTITY="${WANDB_ENTITY:-rm4411-princeton-university}"
export TOKENIZERS_PARALLELISM=false
# All Ray workers use the single prebuilt base venv (no runtime `uv sync` on air-gapped nodes).
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# Slurm sometimes sets a TMPDIR that doesn't exist on the compute node.
if [[ -n "${TMPDIR:-}" && ! -d "$TMPDIR" ]]; then export TMPDIR=/tmp; fi

module load cudatoolkit/12.6 2>/dev/null || true

# Compiled-extension include/lib paths from the venv's bundled CUDA wheels.
NV_PKG="$ROOT/.venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$NV_PKG/cudnn" ]]; then
  export CUDNN_INCLUDE_DIR="$NV_PKG/cudnn/include"
  export CUDNN_LIB_DIR="$NV_PKG/cudnn/lib"
  export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  # All bundled nvidia-wheel includes (cudnn, nccl, curand, cccl, ...) so FlashInfer's JIT nvcc build
  # finds curand_kernel.h etc. on ailab nodes whose system CUDA include is incomplete.
  NV_INC=$(ls -d ${NV_PKG}/*/include 2>/dev/null | tr '\n' ':')
  export CPATH="${NV_INC}${CPATH:+${CPATH}}"
  export CPLUS_INCLUDE_PATH="${NV_INC}${CPLUS_INCLUDE_PATH:+${CPLUS_INCLUDE_PATH}}"
fi

# Unique Ray session root per Slurm job (avoid cross-job Ray auto-attach).
export NRL_RAY_SESSION_DIR="${TMPDIR:-/tmp}/nrl-ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$NRL_RAY_SESSION_DIR" logs/slurm "logs/${RUN_NAME}" "results/${RUN_NAME}"

echo "=== RUN_NAME=$RUN_NAME  FP=$FP  FN=$FN  SEED=$SEED  ROLLOUTS=$ROLLOUTS  MODEL=$MODEL ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --no-sync: use the prebuilt .venv as-is. Without it, uv would try to (re)install deep-ep
# (excluded at build time) and fail on the air-gapped, sm_80 compute node. --offline: never touch network.
uv run --no-sync --offline python examples/run_grpo.py \
  --config "$BASE_CONFIG" \
  policy.model_name="$MODEL" \
  grpo.seed="$SEED" \
  grpo.num_generations_per_prompt="$ROLLOUTS" \
  policy.train_global_batch_size="$GLOBAL_BATCH" \
  ++env.${NOISE_ENV:-math}.fp="$FP" \
  ++env.${NOISE_ENV:-math}.fn="$FN" \
  grpo.val_at_start=true \
  grpo.val_at_end=true \
  logger.log_dir="logs/${RUN_NAME}" \
  logger.wandb.name="$RUN_NAME" \
  logger.wandb.project="RLVR" \
  checkpointing.checkpoint_dir="results/${RUN_NAME}" \
  ${EXTRA_OVERRIDES:-}
