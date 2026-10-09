#!/usr/bin/env bash
set -euo pipefail
swift sft \
  --model /models/Qwen3-1.7B \
  --dataset /opt/agent-scripts/smoke-data.jsonl \
  --split_dataset_ratio 0 \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules q_proj v_proj \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --max_length 256 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --learning_rate 0.0001 \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --dataloader_num_workers 0 \
  --dataloader_persistent_workers false \
  --dataset_num_proc 1 \
  --save_strategy no \
  --eval_strategy no \
  --report_to none \
  --logging_steps 1 \
  --output_dir /workspace/smoke-sft
