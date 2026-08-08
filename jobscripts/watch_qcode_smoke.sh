#!/bin/bash
# Watch the 2 Qwen KodCode smokes; report base->peak validation accuracy (takeoff vs floor).
# Exits when both have >=3 val points (verdict) OR a fail OR ~6h.
set -uo pipefail
names="qcode-1.5B-r16-fp0.0-fn0.0-s1 qcode-3B-r16-fp0.0-fn0.0-s1"
for i in $(seq 1 72); do   # 72 x 5min = 6h
  # any terminal failure?
  bad=$(sacct -S "$(date -d '10 minutes ago' '+%Y-%m-%dT%H:%M')" -n -X -o JobName,State 2>/dev/null | grep -E "qcode" | grep -icE "FAILED|OOM|TIMEOUT|NODE_FAIL")
  ready=0; out=""
  for nm in $names; do
    rid=$(squeue -u aw2418 -h -n "$nm" -o "%A" 2>/dev/null | head -1)
    f=""; [ -n "$rid" ] && f=$(scontrol show job "$rid" 2>/dev/null | grep -oE "StdOut=[^ ]+" | head -1 | cut -d= -f2)
    [ -f "$f" ] || f=$(ls -t logs/slurm/${nm}-*.out 2>/dev/null | head -1)
    vals=$([ -f "$f" ] && grep -aoE "validation/accuracy[^0-9]*[0-9]\.[0-9]+" "$f" 2>/dev/null | grep -oE "[0-9]\.[0-9]+")
    n=$(echo "$vals" | grep -c .)
    base=$(echo "$vals" | head -1); peak=$(echo "$vals" | sort -g | tail -1)
    out="${out}\n  ${nm}: base=${base:-?} peak=${peak:-?} (${n} val pts)"
    [ "${n:-0}" -ge 3 ] && ready=$((ready+1))
  done
  if [ "$ready" -ge 2 ] || [ "${bad:-0}" -gt 0 ]; then
    echo "=== QWEN CODE SMOKE VERDICT (fails=${bad:-0}) ==="; echo -e "$out"
    echo "(peak >> base = takeoff -> gridd-able; peak ~ base = floor -> skip Qwen code)"; exit 0
  fi
  sleep 300
done
echo "TIMEOUT 6h; latest:"; echo -e "$out"
