#!/bin/bash
# Auto-sync offline wandb runs to the online RLVR project so W&B stays the COMPLETE authoritative record.
# Runs every 30 min via a systemd --user timer on the Della login node (needs internet).
#
# Why a STATE FILE, not the .wandb.synced marker: the current wandb `sync` uploads fine (exit 0) but does
# NOT write run-<id>.wandb.synced, so a marker check looks "never synced" and re-uploads forever. Instead
# we record each dir we've successfully synced in logs/.autosync_done and skip it next time. Seeded once
# from the OLD-wandb markers so the already-synced finished grids aren't re-uploaded.
# flock => only one instance ever runs (systemd + any manual run can't race). Junk (<step 90) is skipped.
set -uo pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
source della_env.sh 2>/dev/null || true
export WANDB_MODE=online
LOCK=logs/.autosync.lock; DONE=logs/.autosync_done; ERR=logs/autosync_err.log; LOG=logs/autosync.log
exec 9>"$LOCK"; flock -n 9 || { echo "$(date '+%F %H:%M'): another autosync holds the lock, skip" >>"$LOG"; exit 0; }
# one-time seed: dirs already synced by the old wandb (they carry run-*.wandb.synced) so we don't re-upload them
[ -f "$DONE" ] || find logs/v4* logs/aiOLMO* -name "*.wandb.synced" -printf '%h\n' 2>/dev/null | sort -u > "$DONE"
now=$(date '+%F %H:%M'); synced=0; scanned=0
while IFS= read -r d; do
  scanned=$((scanned+1))
  grep -qxF "$d" "$DONE" && continue                        # already synced (our record)
  ms=$(grep -aoE "Step [0-9]+/[0-9]+" "$d/files/output.log" 2>/dev/null \
       | grep -oE '^Step [0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1)
  [ "${ms:-0}" -ge 90 ] || continue                         # skip junk (MIG fast-fail / early crash)
  if wandb sync "$d" >>"$ERR" 2>&1; then echo "$d" >>"$DONE"; synced=$((synced+1)); fi
done < <(find logs/v4* logs/aiOLMO* -type d -name "offline-run-*" 2>/dev/null)
echo "${now}: scanned=${scanned} newly-synced=${synced}" >>"$LOG"
echo "scanned=${scanned} newly-synced=${synced}"
