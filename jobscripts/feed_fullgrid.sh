#!/bin/bash
# Stream the remaining full-6x6 model-grids into griffith + ailab as the 1100-task submit cap frees.
# Each channel submits its next model-array; retries every 5 min until it fits, then advances.
cd /scratch/gpfs/GRIFFITHS/aw2418/RLVR_noisy_rewards
source della_env.sh 2>/dev/null
MBs="data.train.dataset_name=math data.validation.dataset_name=math data.default.system_prompt_file=examples/prompts/math.txt checkpointing.enabled=false policy.train_micro_batch_size=1 policy.dtensor_cfg.activation_checkpointing=true"
MG="$MBs policy.generation.vllm_cfg.gpu_memory_utilization=0.4"
FSDP="cluster.gpus_per_node=4 policy.dtensor_cfg.tensor_parallel_size=1 policy.generation.vllm_cfg.tensor_parallel_size=4"
N(){ echo $(( $(wc -l < sweeps/$1.txt) - 1 )); }
log(){ echo "[$(date '+%m-%d %H:%M')] $*" >> logs/feeder.log; }

sub_grif_nomig(){ sbatch --job-name=$1 --time=24:00:00 --constraint=nomig --array=0-$(N $1)%16 \
  --export=ALL,PARAMS_FILE=sweeps/$1.txt,EXTRA_OVERRIDES="$MBs" jobscripts/grpo_della.sh 2>&1; }
sub_grif_gemma(){ sbatch --job-name=$1 --time=24:00:00 --constraint=gpu80 --mem=128G --cpus-per-task=16 --array=0-$(N $1)%6 \
  --export=ALL,VLLM_ATTENTION_BACKEND=FLASHINFER,PARAMS_FILE=sweeps/$1.txt,EXTRA_OVERRIDES="$MG" jobscripts/grpo_della.sh 2>&1; }
sub_ailab_fsdp(){ sbatch --job-name=$1 --account=griffith --partition=ailab --time=2-00:00:00 --gres=gpu:4 --cpus-per-task=32 --mem=360G --array=0-$(N $1)%2 \
  --export=ALL,PARAMS_FILE=sweeps/$1.txt,EXTRA_OVERRIDES="$FSDP $MBs" jobscripts/grpo_della.sh 2>&1; }
sub_ailab_gemma(){ sbatch --job-name=$1 --account=griffith --partition=ailab --time=2-00:00:00 --gres=gpu:4 --cpus-per-task=32 --mem=360G --array=0-$(N $1)%2 \
  --export=ALL,VLLM_ATTENTION_BACKEND=FLASHINFER,PARAMS_FILE=sweeps/$1.txt,EXTRA_OVERRIDES="$FSDP $MG" jobscripts/grpo_della.sh 2>&1; }

feed(){ local fn=$1 arr=$2 i=0
  until $fn $arr | grep -q "Submitted batch job"; do
    i=$((i+1)); [ $i -gt 800 ] && { log "$arr GAVE UP after 800 retries"; return 1; }
    sleep 300
  done
  log "$arr SUBMITTED"
}
log "feeder start"
( for spec in "sub_grif_nomig fg_q15" "sub_grif_nomig fg_o1" "sub_grif_gemma fg_g2" "sub_grif_gemma fg_g4"; do feed $spec; done; log "griffith channel done" ) &
( for spec in "sub_ailab_fsdp fg_o7" "sub_ailab_gemma fg_g9"; do feed $spec; done; log "ailab channel done" ) &
wait
log "FEEDER COMPLETE — all 8 full-grid models submitted"
