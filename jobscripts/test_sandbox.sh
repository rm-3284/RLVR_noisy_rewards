#!/bin/bash -l
#SBATCH --job-name=sbxtest
#SBATCH --account=griffith
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards/della_env.sh
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
uv run --no-sync --offline python nemo_rl/environments/test_sandbox_containment.py
