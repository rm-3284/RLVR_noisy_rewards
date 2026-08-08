#!/bin/bash
# Completeness audit of the core master grids (prefix-r{R}-fp{}-fn{}-s{} naming, 240 cells each).
# Three buckets that matter for the beta1 held-out validation (which reads PEAK acc ~step 90):
#   clean  = max step >= M-2         (full epoch)
#   usable = 120 <= max step < M-2   (peak+curve captured; fine for beta1, just not pristine)
#   GAP    = max step < 120 or no dir (never reached peak / never ran -> BLOCKS beta1)
# M (total steps) is read per-cell from its own "Step N/M" line (GSM8K vs MATH may differ slightly).
set -uo pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
OUT=/tmp/claude-356053/core_grid_audit.txt
: > "$OUT"
printf "%-16s %6s %6s %6s %6s\n" "GRID" "clean" "usable" "GAP" "total" | tee -a "$OUT"
for pref in "$@"; do
  clean=0; usable=0; gap=0; gaplist=""
  for r in 8 32 128; do for fp in 0.0 0.15 0.3 0.45; do for fn in 0.0 0.15 0.3 0.45; do for s in 1 2 3 4 5; do
    rn="${pref}-r${r}-fp${fp}-fn${fn}-s${s}"
    line=$(grep -rhaoE "Step [0-9]+/[0-9]+" "logs/$rn" 2>/dev/null | sort -t/ -k1.6 -n | tail -1)
    n=$(echo "$line" | grep -oE '^Step [0-9]+' | grep -oE '[0-9]+')
    m=$(echo "$line" | grep -oE '/[0-9]+$' | tr -d /)
    n=${n:-0}; m=${m:-234}
    if   [ "$n" -ge $((m-2)) ] && [ "$n" -gt 0 ]; then clean=$((clean+1))
    elif [ "$n" -ge 120 ]; then usable=$((usable+1))
    else gap=$((gap+1)); gaplist="${gaplist}${rn}(${n}) "
    fi
  done; done; done; done
  printf "%-16s %6d %6d %6d %6d\n" "$pref" "$clean" "$usable" "$gap" "$((clean+usable+gap))" | tee -a "$OUT"
  echo "  GAPs: ${gaplist:-none}" >> "$OUT"
done
echo "=== full gap lists in $OUT ===" | tee -a "$OUT"
