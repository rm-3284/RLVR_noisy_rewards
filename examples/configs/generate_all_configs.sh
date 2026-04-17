#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$SCRIPT_DIR/config_generator.py"

BATCH_SIZES="16,32,64"
ROLLOUTS="4,8,16,32,64"
FPS="0.0,0.1,0.2,0.3,0.4,0.5"
FNS="0.0,0.1,0.2,0.3,0.4,0.5"
DATASET="gsm8k"

echo "Generating configs for Qwen2.5-0.5B..."
python "$GENERATOR" \
    --batch-sizes "$BATCH_SIZES" \
    --rollouts "$ROLLOUTS" \
    --fps "$FPS" \
    --fns "$FNS" \
    --model-path "/n/fs/vision-mix/rm4411/hf_models/Qwen2.5-0.5B" \
    --dataset "$DATASET" \
    --out-dir "$SCRIPT_DIR/generated_configs_qwen_0.5B_fpfn" \
    --log-dir "logs_0.5B" \
    --run-prefix "grpo0.5B"

echo "Generating configs for Qwen2.5-1.5B..."
python "$GENERATOR" \
    --batch-sizes "$BATCH_SIZES" \
    --rollouts "$ROLLOUTS" \
    --fps "$FPS" \
    --fns "$FNS" \
    --model-path "/n/fs/vision-mix/rm4411/hf_models/Qwen2.5-1.5B" \
    --dataset "$DATASET" \
    --out-dir "$SCRIPT_DIR/generated_configs_qwen1.5B_fpfn" \
    --log-dir "logs_1.5B" \
    --run-prefix "grpo1.5B"

echo "Generating configs for Gemma-3-270M..."
python "$GENERATOR" \
    --batch-sizes "$BATCH_SIZES" \
    --rollouts "$ROLLOUTS" \
    --fps "$FPS" \
    --fns "$FNS" \
    --model-path "/n/fs/vision-mix/rm4411/hf_models/gemma-3-270m" \
    --dataset "$DATASET" \
    --out-dir "$SCRIPT_DIR/generated_configs_gemma3_270m_fpfn" \
    --log-dir "logs_gemma3_270m" \
    --run-prefix "grpo270m"

echo "Generating configs for Gemma-3-1B..."
python "$GENERATOR" \
    --batch-sizes "$BATCH_SIZES" \
    --rollouts "$ROLLOUTS" \
    --fps "$FPS" \
    --fns "$FNS" \
    --model-path "/n/fs/vision-mix/rm4411/hf_models/gemma-3-1b-pt" \
    --dataset "$DATASET" \
    --out-dir "$SCRIPT_DIR/generated_configs_gemma3_1b_fpfn" \
    --log-dir "logs_gemma3_1b" \
    --run-prefix "grpo1b"

echo "Done."
