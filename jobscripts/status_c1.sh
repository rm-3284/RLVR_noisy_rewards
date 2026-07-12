#!/bin/bash -l
# Per-run status for the C1 array. Joins each array index -> run name (sweeps/c1_params.txt)
# with its live/finished state. Writes full table to logs/c1_status.txt; prints a summary.
ARRAY="${1:-10430705}"
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
PARAMS=sweeps/c1_params.txt
OUT=logs/c1_status.txt

declare -A STATE
# live queue states (PENDING/RUNNING) keyed by array index
while read -r idx st; do [ -n "$idx" ] && STATE[$idx]="$st"; done < <(squeue --me -h -r -o "%K %T" 2>/dev/null | grep -E "^[0-9]+ ")
# finished states from sacct (override; COMPLETED/FAILED/etc) keyed by array index
while read -r jid st; do
  i="${jid##*_}"; [[ "$i" =~ ^[0-9]+$ ]] && STATE[$i]="$st"
done < <(sacct -j "$ARRAY" -n -X -o "JobID,State" 2>/dev/null | awk '{print $1, $2}')

: > "$OUT"
i=0
while read -r name fp fn seed rest; do
  st="${STATE[$i]:-UNKNOWN}"
  printf "%3d  %-12s  %s\n" "$i" "$st" "$name" >> "$OUT"
  i=$((i+1))
done < "$PARAMS"

echo "=== C1 per-run status ($(date '+%H:%M')) — full table: $OUT ==="
echo "--- counts by state ---"; awk '{print $2}' "$OUT" | sort | uniq -c
echo "--- RUNNING ---"; grep " RUNNING " "$OUT" | awk '{print $3}' | tr '\n' ' '; echo
fin=$(grep -E " COMPLETED | FAILED | TIMEOUT | OUT_OF_ | CANCELLED " "$OUT")
[ -n "$fin" ] && { echo "--- FINISHED ---"; echo "$fin"; }
