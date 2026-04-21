#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-Qwen/Qwen3-32B}"
ADAPTER="${ADAPTER:-Judge-lora/outputs/qwen3-32b-judge-lora/final}"
OUT="${OUT:-Judge-lora/outputs/qwen3-32b-judge-merged}"

python Judge-lora/merge_lora.py \
    --base "${BASE}" \
    --adapter "${ADAPTER}" \
    --out "${OUT}"
