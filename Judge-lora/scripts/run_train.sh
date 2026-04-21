#!/usr/bin/env bash
# Launch LoRA training with DeepSpeed ZeRO-3 on 4 GPUs. Run from the repo root:
#   bash Judge-lora/scripts/run_train.sh
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# Helps with fragmentation at long sequence lengths.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="${CONFIG:-Judge-lora/configs/qwen3_32b_judge.yaml}"
NPROC="${NPROC:-4}"

accelerate launch \
    --num_processes "${NPROC}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    Judge-lora/train.py \
    --config "${CONFIG}"
