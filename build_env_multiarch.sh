#!/bin/bash -l
# Build a SECOND venv at $SCR/venv-multiarch compiled for BOTH A100 (sm_80) and H100/H200 (sm_90),
# so PLI/AI-Lab jobs can run from it. Does NOT touch the live ./.venv used by the running C1 jobs.
# --refresh-package forces the source-built extensions to recompile with the new arch list
# (otherwise uv reuses the cached sm_80-only wheels and the result is still A100-only).
set -o pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards || exit 1
source della_env.sh
module load cudatoolkit/12.6 2>/dev/null || true

export TORCH_CUDA_ARCH_LIST="8.0 9.0"      # A100 + H100/H200
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export UV_PROJECT_ENVIRONMENT="$SCR/venv-multiarch"

# cuDNN/NCCL headers for the TE build (same headers regardless of arch; reuse the live .venv's copy).
NV="$ROOT/.venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$NV" ]]; then
  for d in "$NV"/*/include; do [[ -d "$d" ]] && export CPATH="${d}:${CPATH:-}"; done
  for d in "$NV"/*/lib;     do [[ -d "$d" ]] && export LIBRARY_PATH="${d}:${LIBRARY_PATH:-}"; done
  export CUDNN_PATH="$NV/cudnn"
fi

# uv's cache key does NOT include TORCH_CUDA_ARCH_LIST, so it will reuse the sm_80-only wheels
# unless we evict them. Clear the cached build artifacts for the compiled extensions, then force
# a reinstall so they recompile from source for sm_80+sm_90.
echo "===== $(date) : clearing cached wheels for compiled extensions ====="
uv cache clean transformer-engine-torch nv-grouped-gemm mamba-ssm causal-conv1d transformer-engine 2>&1 | tail -3

echo "===== $(date) : building multiarch venv at $UV_PROJECT_ENVIRONMENT (sm_80+sm_90) ====="
uv sync --frozen --extra automodel --extra vllm \
  --no-install-package deep-ep --no-install-package deep-gemm \
  --reinstall-package transformer-engine-torch \
  --reinstall-package nv-grouped-gemm \
  --reinstall-package mamba-ssm \
  --reinstall-package causal-conv1d 2>&1
rc=$?
echo "===== $(date) : multiarch uv sync exit code = $rc ====="
exit $rc
