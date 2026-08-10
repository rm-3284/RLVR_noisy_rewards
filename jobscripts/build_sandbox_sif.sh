#!/bin/bash
# Build the offline code-sandbox SIF (apptainer). Run on a node WITH internet (login), once per cluster.
# code_verify.py reads it from $CODE_SANDBOX_SIF (default below). Compute nodes are air-gapped, so it
# must live on shared storage. Apptainer runs setuid -> works even where user namespaces are disabled.
set -euo pipefail
OUT="${1:-/scratch/gpfs/GRIFFITHS/aw2418/code_sandbox.sif}"
apptainer build --force "$OUT" docker://python:3.11-slim
echo "built: $OUT  (set CODE_SANDBOX_SIF to override the path)"
