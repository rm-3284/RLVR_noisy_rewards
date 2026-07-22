#!/bin/bash -l
#SBATCH --job-name=rlvri
#SBATCH --account=allcs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00          # ionic allows 7 days; generous so r128 finishes without the wall
#SBATCH --mail-type=fail
#SBATCH --mail-user=aw2418@princeton.edu
#SBATCH --output=/n/fs/scratch/aw2418/RLVR_noisy_rewards/logs/slurm/%x-%j.out
#SBATCH --error=/n/fs/scratch/aw2418/RLVR_noisy_rewards/logs/slurm/%x-%j.out
#
# Parameterized GRPO run on the CS ionic cluster. Mirror of grpo_della.sh.
# Reads a sweep line via PARAMS_FILE + SLURM_ARRAY_TASK_ID, or runs standalone (smoke).

set -euo pipefail
source /n/fs/scratch/aw2418/RLVR_noisy_rewards/ionic_env.sh

# ---- array mode: RUN_NAME FP FN SEED ROLLOUTS MODEL ----
if [[ -n "${PARAMS_FILE:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAMS_FILE")
  read -r RUN_NAME FP FN SEED ROLLOUTS MODEL <<< "$line"
fi

FP="${FP:-0.0}"
FN="${FN:-0.0}"
SEED="${SEED:-42}"
ROLLOUTS="${ROLLOUTS:-8}"
MODEL="${MODEL:-/n/fs/scratch/aw2418/hf_models/Qwen2.5-3B}"
RUN_NAME="${RUN_NAME:-ionic-smoke}"
BASE_CONFIG="${BASE_CONFIG:-examples/configs/grpo_gsm8k_1B_rollout32_batch32.yaml}"
NUM_PROMPTS=32
GLOBAL_BATCH=$(( NUM_PROMPTS * ROLLOUTS ))

# compute nodes have internet: allow HF dataset download; keep wandb offline unless overridden
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-rm4411-princeton-university}"
export TOKENIZERS_PARALLELISM=false
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

if [[ -n "${TMPDIR:-}" && ! -d "$TMPDIR" ]]; then export TMPDIR=/tmp; fi

# Compiled-extension include/lib paths from the venv's bundled CUDA wheels (FlashInfer JIT etc.)
NV_PKG="$ROOT/.venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$NV_PKG/cudnn" ]]; then
  export CUDNN_INCLUDE_DIR="$NV_PKG/cudnn/include"
  export CUDNN_LIB_DIR="$NV_PKG/cudnn/lib"
  export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  NV_INC=$(ls -d ${NV_PKG}/*/include 2>/dev/null | tr '\n' ':')
  export CPATH="${NV_INC}${CPATH:+${CPATH}}"
  export CPLUS_INCLUDE_PATH="${NV_INC}${CPLUS_INCLUDE_PATH:+${CPLUS_INCLUDE_PATH}}"
fi

export NRL_RAY_SESSION_DIR="${TMPDIR:-/tmp}/nrl-ray-${SLURM_JOB_ID:-$$}"
mkdir -p "$NRL_RAY_SESSION_DIR" logs/slurm "logs/${RUN_NAME}" "results/${RUN_NAME}"

echo "=== RUN_NAME=$RUN_NAME  FP=$FP  FN=$FN  SEED=$SEED  ROLLOUTS=$ROLLOUTS  MODEL=$MODEL  on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --no-sync: use the prebuilt .venv as-is. NO --offline: compute nodes have internet (dataset dl).
uv run --no-sync python examples/run_grpo.py \
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
