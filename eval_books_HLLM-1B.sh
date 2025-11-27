#!/bin/bash

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

cd code

python3 main.py \
    --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
    --loss nce \
    --epochs 5 \
    --dataset amazon_books \
    --train_batch_size 2 \
    --MAX_TEXT_LENGTH 256 \
    --MAX_ITEM_LIST_LENGTH 10 \
    --checkpoint_dir /root/autodl-tmp/HLLM/checkpoints \
    --optim_args.learning_rate 1e-4 \
    --item_pretrain_dir "$MODEL_DIR" \
    --user_pretrain_dir "$MODEL_DIR" \
    --text_path /root/autodl-tmp/HLLM/information \
    --text_keys '[\"title\",\"description\"]' \
    --val_only false