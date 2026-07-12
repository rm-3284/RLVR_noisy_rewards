#!/bin/bash -l
# Relaunch the two ailab APPS sweeps with the CORRECTED reward-env override.
#
# BUG (found 2026-07-07): the original ai3Bapps / aiAPPS05 launches loaded APPS data
# (data.*.dataset_name=apps) but did NOT set data.default.env_name=code_verify. The base
# config default env is "math", so APPS code solutions were graded by the MATH verifier
# (\boxed{} extraction) -> reward exactly 0.000 for every step/seed. The working 1.5B smoke
# (appssm2) differed ONLY by these three overrides; adding them replicates that known-good recipe.
#
# Usage:
#   jobscripts/relaunch_apps_ailab.sh --dry-run   # print the two sbatch cmds, submit nothing
#   jobscripts/relaunch_apps_ailab.sh             # submit both corrected arrays
set -euo pipefail
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
source della_env.sh

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# Corrected overrides = original APPS overrides + the 3 that route grading to the code verifier.
OVERRIDES="data.train.dataset_name=apps data.validation.dataset_name=apps data.validation.split=test \
data.default.env_name=code_verify data.default.system_prompt_file=null +env.code_verify.num_workers=8 \
policy.train_micro_batch_size=1 policy.dtensor_cfg.activation_checkpointing=true \
policy.generation.vllm_cfg.gpu_memory_utilization=0.35 grpo.max_num_epochs=2 \
grpo.max_num_steps=120 checkpointing.enabled=false"

# name            params_file                     mem    array-throttle (matches original)
SWEEPS=(
  "ai3Bapps  sweeps/ai3Bapps_params.txt  180G  24"
  "aiAPPS05  sweeps/aiAPPS05_params.txt  120G  10"
)

for row in "${SWEEPS[@]}"; do
  read -r NAME PARAMS MEM THROTTLE <<< "$row"
  N=$(wc -l < "$PARAMS")
  CMD=(sbatch --job-name="$NAME" --account=griffith --partition=ailab --time=2-00:00:00
       --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem="$MEM"
       --array=0-$((N - 1))%${THROTTLE}
       --export=ALL,NOISE_ENV=code_verify,PARAMS_FILE="$PARAMS",EXTRA_OVERRIDES="$OVERRIDES"
       jobscripts/grpo_della.sh)
  if [[ -n "$DRY" ]]; then
    echo "[dry-run] $NAME ($N cells, throttle $THROTTLE):"; printf '  %q' "${CMD[@]}"; echo; echo
  else
    "${CMD[@]}"
  fi
done
