#!/bin/bash
#SBATCH --job-name=grpo_math  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=64G                  # Memory per node
#SBATCH --time=20:00:00            # Time limit (1 hour)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/output.log
#SBATCH --error=logs/%j/error.log
#SBATCH --partition=all

export HF_HOME=/n/fs/vision-mix/rm4411/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=/n/fs/vision-mix/rm4411/huggingface
export TORCH_HOME=/n/fs/vision-mix/rm4411/torch
export XDG_CACHE_HOME=/n/fs/vision-mix/rm4411/cache
export CACHE_DIR=/n/fs/vision-mix/rm4411/cache
ROOT=/n/fs/vision-mix/rm4411/RLVR_noisy_rewards
cd "$ROOT" || exit 1
# Do not use PYTHONPATH="$ROOT:$PYTHONPATH". If login rc adds e.g. ~/.local/lib/python3.11/site-packages,
# that path is searched *before* the venv and shadows packages (huggingface_hub 1.7.2 vs 0.34 in .venv).
export PYTHONPATH="$ROOT"
export PYTHONNOUSERSITE=1
export LD_PRELOAD="/n/fs/vision-mix/rm4411/conda_envs/understand_bias/lib/libstdc++.so.6"

module load cudatoolkit/12.6

NV_PKG="$ROOT/.venv/lib/python3.12/site-packages/nvidia"
export CUDNN_INCLUDE_DIR="$NV_PKG/cudnn/include"
export CUDNN_LIB_DIR="$NV_PKG/cudnn/lib"
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
# torch/cpp_extension + ninja do not add CUDNN_INCLUDE_DIR / NCCL to -I; use compiler search path.
export CPATH="${CUDNN_INCLUDE_DIR}:${NV_PKG}/nccl/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CUDNN_INCLUDE_DIR}:${NV_PKG}/nccl/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

# W&B: set before sbatch, e.g. `export WANDB_API_KEY=...` (do not store secrets in this file).
# If this key was ever committed, rotate it at https://wandb.ai/settings

# Pin the env to uv.lock and drop stray packages (e.g. huggingface-hub 1.x) before running.
# On air-gapped nodes, run `uv sync --frozen --exact` once on a machine with cache/network, or use:
#   exec "$ROOT/.venv/bin/python" "$ROOT/examples/run_grpo.py" ...
# DTensorPolicyWorkerV2 uses PY_EXECUTABLES.AUTOMODEL (nemo-automodel, TE, etc.); keep lock + extras in sync.
uv sync --frozen --exact --extra automodel
uv run --frozen --extra automodel python examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml