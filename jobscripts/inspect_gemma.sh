#!/bin/bash -l
#SBATCH --job-name=geminspect
#SBATCH --account=griffith
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --constraint=gpu80
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards/della_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASHINFER
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
uv run --no-sync --offline python notebook/inspect_gemma_fmt.py
