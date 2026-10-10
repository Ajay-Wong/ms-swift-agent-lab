#!/usr/bin/env bash
# Run inside the training container. Extra CLI arguments override defaults.
set -euo pipefail
DATA_DIR="${DATA_DIR:-/workspace/shopping/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/shopping/sft}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3-1.7B}"
for file in train.sft.jsonl val.sft.jsonl manifest.json; do
  test -f "$DATA_DIR/$file" || { echo "Missing $DATA_DIR/$file; generate the dataset first." >&2; exit 1; }
done
swift sft \
  --model "$MODEL_DIR" \
  --dataset "$DATA_DIR/train.sft.jsonl" \
  --val_dataset "$DATA_DIR/val.sft.jsonl" \
  --split_dataset_ratio 0 \
  --template qwen3 \
  --agent_template hermes \
  --enable_thinking false \
  --loss_scale default \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules q_proj k_proj v_proj o_proj \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_length 2048 \
  --truncation_strategy delete \
  --strict true \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --learning_rate 0.0001 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 0 \
  --dataloader_persistent_workers false \
  --dataset_num_proc 1 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 2 \
  --eval_strategy steps \
  --eval_steps 50 \
  --logging_steps 5 \
  --report_to none \
  --seed 42 \
  --output_dir "$OUTPUT_DIR" \
  "$@"
