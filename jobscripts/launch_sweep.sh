#!/bin/bash -l
# Generate a sweep param file and submit it as a single SLURM array job.
#
# Usage:
#   jobscripts/launch_sweep.sh c1            # C1 dense 6x6 (fp,fn) surface, 1.5B, r=32, 5 seeds = 180 runs
#   jobscripts/launch_sweep.sh c1 --dry-run  # write the param file + print the sbatch cmd, do not submit
#
# Concurrency is capped with the array %N suffix to stay under the gpu-short per-user job limit.

set -euo pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
source della_env.sh

WHICH="${1:-c1}"
DRY=""
[[ "${2:-}" == "--dry-run" ]] && DRY=1

mkdir -p sweeps logs/slurm
PARAMS="sweeps/${WHICH}_params.txt"
: > "$PARAMS"

case "$WHICH" in
  c1)
    MODEL="/scratch/gpfs/GRIFFITHS/aw2418/hf_models/Qwen2.5-1.5B"; ROLLOUTS=32
    NOISE="0.0 0.1 0.2 0.3 0.4 0.5"
    SEEDS="1 2 3 4 5"
    for fp in $NOISE; do for fn in $NOISE; do for s in $SEEDS; do
      name="c1-1.5B-fp${fp}-fn${fn}-r${ROLLOUTS}-s${s}"
      echo "$name $fp $fn $s $ROLLOUTS $MODEL" >> "$PARAMS"
    done; done; done
    MAXCONC=44   # gpu-short per-user hard cap is 44; this is the practical max concurrency
    ;;
  c1r8)
    # Two-r gate hedge: gate-critical cells only, at r=8, to test whether the asymmetry verdict is r-specific.
    MODEL="/scratch/gpfs/GRIFFITHS/aw2418/hf_models/Qwen2.5-1.5B"; ROLLOUTS=8
    CELLS="0.0:0.0 0.0:0.3 0.3:0.0 0.3:0.3"   # clean, FN anchor, FP anchor, symmetric
    SEEDS="1 2 3 4 5"
    for c in $CELLS; do fp="${c%%:*}"; fn="${c##*:}"; for s in $SEEDS; do
      name="c1r8-1.5B-fp${fp}-fn${fn}-r${ROLLOUTS}-s${s}"
      echo "$name $fp $fn $s $ROLLOUTS $MODEL" >> "$PARAMS"
    done; done
    MAXCONC=10   # trickle alongside the r=32 array; the 44-GPU per-user cap is shared between them
    ;;
  *)
    echo "Unknown sweep '$WHICH'"; exit 1 ;;
esac

N=$(wc -l < "$PARAMS")
echo "Wrote $N jobs to $PARAMS"
# Gate metrics all come from W&B; model checkpoints would cost ~1.6TB and blow the shared GRIFFITHS
# fileset (only ~1.6TB free lab-wide). Disable checkpointing for the sweep.
SWEEP_OVERRIDES="checkpointing.enabled=false"
# --constraint=nomig: vLLM can't parse MIG device UUIDs (int('MIG-...') crashes), so exclude MIG slices.
SBATCH_CMD=(sbatch --job-name="$WHICH" --qos=gpu-short --time=16:00:00 --constraint=nomig
            --array=0-$((N - 1))%${MAXCONC}
            --export=ALL,PARAMS_FILE="$PARAMS",EXTRA_OVERRIDES="$SWEEP_OVERRIDES" jobscripts/grpo_della.sh)

if [[ -n "$DRY" ]]; then
  echo "[dry-run] would submit:"; printf '  %q' "${SBATCH_CMD[@]}"; echo
  echo "--- first 3 param lines ---"; head -3 "$PARAMS"
else
  "${SBATCH_CMD[@]}"
fi
