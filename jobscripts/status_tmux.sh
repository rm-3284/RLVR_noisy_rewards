#!/bin/bash
# Start (or reuse) the persistent 'rlvr' status tmux on THIS login node.
#   view:   tmux attach -t rlvr        (detach: Ctrl-b then d)
#   restart after a reboot/maintenance: just run this script again.
# NOTE: tmux is per-login-node. This runs on whatever node you launch it from; attach from the SAME node
#       (ssh della9 first if you land elsewhere). Slurm state is cluster-wide so the numbers are the same.
S=/scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards/jobscripts/status.sh
if tmux has-session -t rlvr 2>/dev/null; then
  echo "'rlvr' already running on $(hostname -s).  attach:  tmux attach -t rlvr"
else
  tmux new-session -d -s rlvr "while true; do clear; bash $S 2>&1; sleep 30; done"
  echo "started 'rlvr' on $(hostname -s).  attach:  tmux attach -t rlvr   (detach: Ctrl-b d)"
fi
