#!/usr/bin/env bash
# DPO training on a tiny preferences dataset.
# Usage: ./scripts/run_dpo.sh [data_path] [output_dir]
set -euo pipefail

DATA_PATH=${1:-data/sample_dpo.jsonl}
OUT=${2:-output/dpo}

python -m cli.train_dpo \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT" \
    --max_examples 200 \
    --max_steps 50 \
    --per_device_batch_size 2 \
    --grad_accum_steps 8 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --use_lora \
    --ref_device cpu \
    --log_every 5 \
    --save_every 50
