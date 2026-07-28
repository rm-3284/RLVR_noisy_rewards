#!/bin/bash
# Robust liveness / wedge check that WORKS FOR ARRAY JOBS.
# The old cron step guessed the log name from `${jobname}-${id}.out`, which is unreliable for array
# cells (they share a jobname and the log is named by the component sub-job id). This resolves each
# running job's ACTUAL StdOut path via `scontrol`, so per-array-cell wedges are caught.
# A "wedge" = Slurm says RUNNING but the log hasn't advanced in >25 min (Ray/vLLM hang, 0% GPU).
set -uo pipefail
U=aw2418
now=$(date +%s); nrun=0; nlive=0
for id in $(squeue -u "$U" -h -r -t R -o "%A"); do
  nrun=$((nrun+1))
  f=$(scontrol show job "$id" 2>/dev/null | grep -oE "StdOut=[^ ]+" | cut -d= -f2)
  [ -f "$f" ] || { echo "NOLOG $id"; continue; }
  age=$(( (now - $(stat -c %Y "$f")) / 60 ))
  step=$(grep -aoiE "Step [0-9]+" "$f" 2>/dev/null | tail -1)
  nm=$(squeue -u "$U" -h -j "$id" -o "%j" 2>/dev/null)
  if [ "$age" -gt 25 ]; then
    echo "WEDGE? ${nm}/${id} log ${age}min stale, step=${step:-NONE}  (StdOut=$f)"
  elif [ -n "$step" ]; then
    nlive=$((nlive+1))
  fi
done
echo "RUNNING=$nrun VERIFIED_LIVE=$nlive"
