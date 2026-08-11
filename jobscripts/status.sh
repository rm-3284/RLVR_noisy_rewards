#!/bin/bash
# One-screen RLVR study status. Refreshed by the 'rlvr' tmux session. Slurm-only (no venv) so it's robust.
U=aw2418
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards 2>/dev/null || true
now=$(date +%s)
echo "==== RLVR STATUS   $(date '+%a %b %d %H:%M:%S')   login=$(hostname -s) ===="
# maintenance reservation (if any active/upcoming)
res=$(scontrol show reservation 2>/dev/null | grep -oE "StartTime=2[0-9T:-]+ EndTime=2[0-9T:-]+" | head -1)
[ -n "$res" ] && echo "!! MAINT: $res"
R=$(squeue -u $U -h -r -t R 2>/dev/null | wc -l)
P=$(squeue -u $U -h -r -t PD 2>/dev/null | wc -l)
G=$(squeue -u $U -h -r -t R -o "%b" 2>/dev/null | grep -oE "gpu:[0-9]+" | cut -d: -f2 | paste -sd+ | bc 2>/dev/null)
echo "RUNNING=$R  (${G:-0} GPUs)   PENDING=$P"
echo
echo "-- by grid (running / pending) --"
{ squeue -u $U -h -r -t R -o "%j" 2>/dev/null | sed -E 's/-fp.*//;s/-r[0-9]+.*//' | sort | uniq -c | sed 's/$/ RUN/'
  squeue -u $U -h -r -t PD -o "%j" 2>/dev/null | sed -E 's/-fp.*//;s/-r[0-9]+.*//' | sort | uniq -c | sed 's/$/ pend/'; } \
  | awk '{c=$1;st=$3;g=$2; if(st=="RUN")run[g]=c; else pend[g]=c; seen[g]=1}
         END{for(k in seen)printf "  %-14s %4d run / %-4d pend\n", k, run[k]+0, pend[k]+0}' | sort
echo
echo "-- running cells: step progress (avg) --"
for rid in $(squeue -u $U -h -r -t R -o "%A" 2>/dev/null); do
  info=$(scontrol show job "$rid" 2>/dev/null)
  nm=$(echo "$info"|grep -oE "JobName=[^ ]+"|head -1|cut -d= -f2)
  f=$(echo "$info"|grep -oE "StdOut=[^ ]+"|head -1|cut -d= -f2)
  st=""; [ -f "$f" ] && st=$(grep -aoE "Step [0-9]+/[0-9]+" "$f" 2>/dev/null|tail -1|grep -oE "^Step [0-9]+"|grep -oE "[0-9]+")
  a=99; [ -f "$f" ] && a=$(( (now - $(stat -c %Y "$f"))/60 ))
  echo "$nm ${st:-0} $a"
done | awk '{c[$1]++;s[$1]+=$2;if($2>mx[$1])mx[$1]=$2;if($3>old[$1])old[$1]=$3}
            END{for(k in c)printf "  %-14s %d cells, ~step %d (max %d), oldest log %dmin\n",k,c[k],s[k]/c[k],mx[k],old[k]}'
[ "$R" -eq 0 ] && echo "  (none running)"
echo
echo "-- last 6h terminal outcomes --"
sacct -u $U -S "$(date -d '6 hours ago' '+%Y-%m-%dT%H:%M')" -n -X -o State%20 2>/dev/null \
  | grep -oiE "COMPLETED|TIMEOUT|OUT_OF_MEMORY|FAILED|NODE_FAIL" | sort | uniq -c | sed 's/^/  /'
echo
echo "(auto-refreshes ~30s | detach: Ctrl-b then d)"
