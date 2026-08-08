#!/bin/bash
# Audit the full 240-cell Qwen-3B (v4q3) grid: for each cell find the MAX step reached across all
# attempts (grep the wandb offline output.log via glob, not recursive -> fast). Classify:
#   clean   = max step >= 234 (full epoch, no timeout)
#   redo    = max step  < 234 (timeout / incomplete / MIG-gap / never-run)
# Excludes cells CURRENTLY RUNNING on Della (they'll finish there) from the redo list.
# Emits the CAIS sweep lines (RUN_NAME FP FN SEED ROLLOUTS MODEL) for the redo set.
set -uo pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
MODEL=/scratch/gpfs/GRIFFITHS/aw2418/hf_models/Qwen2.5-3B
OUT=/tmp/claude-356053/v4q3_audit.txt
SWEEP=sweeps/cais_v4q3_remainder.txt
: > "$OUT"; : > "$SWEEP"

# cells currently running on Della (exclude from redo)
declare -A RUN
for rid in $(squeue -u aw2418 -h -r -t R -o "%A %j" 2>/dev/null | awk '$2 ~ /v4q3/{print $1}'); do
  f=$(scontrol show job "$rid" 2>/dev/null | grep -oE "StdOut=[^ ]+" | head -1 | cut -d= -f2)
  rn=$([ -f "$f" ] && grep -aoE "wandb.name=v4q3-[a-z0-9.-]+" "$f" 2>/dev/null | head -1 | cut -d= -f2)
  [ -n "$rn" ] && RUN[$rn]=1
done

clean=0; redo=0; running=0
for r in 8 32 128; do for fp in 0.0 0.15 0.3 0.45; do for fn in 0.0 0.15 0.3 0.45; do for s in 1 2 3 4 5; do
  rn="v4q3-r${r}-fp${fp}-fn${fn}-s${s}"
  # recursive grep over the whole run dir (the glob path missed re-attempt/alt log locations -> false redos)
  ms=$(grep -rhaoE "Step [0-9]+/234" logs/"$rn" 2>/dev/null \
       | grep -oE '^Step [0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1)
  ms=${ms:-0}
  if [ "$ms" -ge 234 ]; then echo "CLEAN  $rn $ms" >>"$OUT"; clean=$((clean+1))
  elif [ -n "${RUN[$rn]:-}" ]; then echo "RUNNING $rn $ms (Della-finishing)" >>"$OUT"; running=$((running+1))
  else echo "REDO   $rn $ms" >>"$OUT"; redo=$((redo+1))
       echo "$rn $fp $fn $s $r $MODEL" >>"$SWEEP"
  fi
done; done; done; done
echo "=== AUDIT DONE: clean=$clean redo=$redo running(Della)=$running total=$((clean+redo+running)) ===" | tee -a "$OUT"
echo "redo breakdown by rollout:" | tee -a "$OUT"
awk '{print $5}' "$SWEEP" | sort | uniq -c | tee -a "$OUT"
echo "CAIS sweep written: $SWEEP ($(wc -l <"$SWEEP") cells)" | tee -a "$OUT"
