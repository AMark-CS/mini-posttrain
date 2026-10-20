#!/usr/bin/env bash
# SFT training on a tiny dataset.
# Usage: ./scripts/run_sft.sh [data_path] [output_dir]
set -euo pipefail

DATA_PATH=${1:-data/sample_sft.jsonl}
OUT=${2:-output/sft}

python -m cli.train_sft \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT" \
    --max_examples 200 \
    --max_steps 50 \
    --per_device_batch_size 4 \
    --grad_accum_steps 4 \
    --learning_rate 5e-5 \
    --use_lora \
    --log_every 5 \
    --save_every 50
