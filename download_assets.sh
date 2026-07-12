#!/bin/bash -l
# Pre-download models + dataset into HF_HOME on scratch (run on login/vis node WITH internet).
# Compute nodes are air-gapped, so everything must be cached here first. Logs to download_assets.log.
set -o pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards || exit 1
source della_env.sh
mkdir -p "$HF_HOME"

DL() { uvx --from huggingface_hub hf download "$@"; }

echo "===== $(date) : Qwen2.5-0.5B ====="
DL Qwen/Qwen2.5-0.5B 2>&1 || { echo "FAIL 0.5B"; exit 1; }
echo "===== $(date) : Qwen2.5-1.5B ====="
DL Qwen/Qwen2.5-1.5B 2>&1 || { echo "FAIL 1.5B"; exit 1; }
echo "===== $(date) : openai/gsm8k (dataset) ====="
DL openai/gsm8k --repo-type dataset 2>&1 || { echo "FAIL gsm8k"; exit 1; }
echo "===== $(date) : DONE all downloads ====="
