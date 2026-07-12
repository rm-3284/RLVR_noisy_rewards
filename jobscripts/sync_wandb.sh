#!/bin/bash -l
# Upload FINISHED offline W&B runs to wandb.ai, once each. Run on the login node (has internet).
# Skips runs already uploaded (marker file) and runs still being written (mtime < 180s = still training).
# Requires `wandb login` done once.
set -uo pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
source della_env.sh
unset WANDB_MODE   # sync needs online
export WANDB_ENTITY="${WANDB_ENTITY:-rm4411-princeton-university}"

MARK=logs/.wandb_uploaded
touch "$MARK"
now=$(date +%s)
synced=0; skipped=0; running=0

while IFS= read -r d; do
  [ -z "$d" ] && continue
  if grep -qxF "$d" "$MARK"; then skipped=$((skipped+1)); continue; fi
  newest=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  if [ -n "$newest" ] && [ $((now - newest)) -lt 180 ]; then running=$((running+1)); continue; fi  # still training
  if ./.venv/bin/wandb sync "$d" 2>&1 | grep -q "done\."; then
    echo "$d" >> "$MARK"; synced=$((synced+1))
  else
    echo "WARN: sync failed for $d"
  fi
done < <(find logs -type d -name 'offline-run-*' 2>/dev/null | sort)

echo "synced $synced new finished runs, skipped $skipped already-uploaded, $running still training"
