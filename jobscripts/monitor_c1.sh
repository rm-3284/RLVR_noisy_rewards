#!/bin/bash -l
# One C1 progress snapshot. Usage: bash jobscripts/monitor_c1.sh [ARRAY_ID]
# Prints state counts, failures, effective concurrency, recent finishers; appends to logs/c1_progress.log.
ARRAY="${1:-10430705}"
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
PROG=logs/c1_progress.log

now=$(date '+%Y-%m-%d %H:%M')
# live queue state for this array
run=$(squeue --me -h -r -t RUNNING  -o "%i" | grep -c "^${ARRAY}_") 2>/dev/null
pend=$(squeue --me -h -r -t PENDING -o "%i" | grep -c "^${ARRAY}_") 2>/dev/null
# finished-state accounting from sacct (one row per task; ignore .batch/.extern subrows)
mapfile -t done < <(sacct -j "$ARRAY" -n -X -o "State" 2>/dev/null | awk '{print $1}')
comp=$(printf '%s\n' "${done[@]}" | grep -c "COMPLETED")
fail=$(printf '%s\n' "${done[@]}" | grep -cE "FAILED|OOM|TIMEOUT|CANCELLED|NODE_FAIL")

line="[$now] running=$run pending=$pend completed=$comp failed=$fail / 180"
echo "$line"; echo "$line" >> "$PROG"

if [ "$fail" -gt 0 ]; then
  echo "  FAILED tasks:"; sacct -j "$ARRAY" -n -X -o "JobID,State,Elapsed" 2>/dev/null | grep -E "FAILED|OOM|TIMEOUT|NODE_FAIL" | head -10
fi
# step-time pulse from the most recent running log
latest=$(ls -t logs/slurm/c1-*.out 2>/dev/null | head -1)
[ -n "$latest" ] && echo "  pulse ($(basename "$latest")): $(grep -E 'Total step time' "$latest" | tail -1)"
