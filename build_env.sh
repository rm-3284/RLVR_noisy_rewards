#!/bin/bash -l
# One-shot env build on a node WITH internet (login/vis). Logs to build_env.log.
set -o pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards || exit 1
source della_env.sh

echo "===== $(date) : git submodule init ====="
git submodule update --init --recursive 2>&1 || { echo "SUBMODULE FAIL"; exit 1; }

echo "===== $(date) : module load cudatoolkit/12.6 ====="
module load cudatoolkit/12.6 2>&1 || echo "WARN: module load failed (ok if uv brings its own toolkit)"

# Source-built CUDA extensions (nv-grouped-gemm, TE, ...) probe a live GPU unless the target
# arch is given. The login node has no GPU -> set arch explicitly (A100 = sm_80). Add 9.0 for H100/pli.
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-8}"      # be polite on the shared login node
export NVCC_THREADS="${NVCC_THREADS:-4}"

# transformer-engine et al. compile against cuDNN/NCCL/cuBLAS headers that ship in the venv's
# nvidia-* wheels but are not on the compiler search path. Add every nvidia/*/include + lib.
NV="$ROOT/.venv/lib/python3.12/site-packages/nvidia"
if [[ -d "$NV" ]]; then
  for d in "$NV"/*/include;  do [[ -d "$d" ]] && export CPATH="${d}:${CPATH:-}"; done
  for d in "$NV"/*/lib;      do [[ -d "$d" ]] && { export LIBRARY_PATH="${d}:${LIBRARY_PATH:-}"; export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; }; done
  export CUDNN_PATH="$NV/cudnn"
fi
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS  CUDA_HOME=${CUDA_HOME:-unset}  CUDNN_PATH=${CUDNN_PATH:-unset}"

# One fat base venv with BOTH training (automodel) and generation (vllm) extras, so every Ray worker
# can run from it via NEMO_RL_PY_EXECUTABLES_SYSTEM=1 (no runtime `uv sync` on air-gapped compute nodes).
# Skip deep-ep / deep-gemm: both are Hopper(sm_90)-only and unused for A100 bf16.
echo "===== $(date) : uv sync --frozen --extra automodel --extra vllm (skip deep-ep, deep-gemm) ====="
uv sync --frozen --extra automodel --extra vllm \
  --no-install-package deep-ep --no-install-package deep-gemm 2>&1
rc=$?
echo "===== $(date) : uv sync exit code = $rc ====="
exit $rc
