#!/bin/bash
# Compute-node re-verify of the code sandbox (Apptainer, cgroup flags dropped). Run via sbatch.
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
source della_env.sh 2>/dev/null || true
echo "NODE=$(hostname)"
echo "backend: $(python -c 'import nemo_rl.environments.code_verify as m; print("APPTAINER" if m._HAVE_APPTAINER else "bwrap/none"); print(" ".join(m._APPTAINER))' 2>/dev/null)"
python nemo_rl/environments/test_sandbox_containment.py 2>&1 | grep -E "\[PASS\]|\[FAIL\]|CONTAINMENT"
