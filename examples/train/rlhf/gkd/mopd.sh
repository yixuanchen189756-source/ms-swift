#!/usr/bin/env bash

# Example MOPD run. Samples should contain a task column with one of:
# reasoning, codegen, agent. A sample-level teacher_id still overrides the
# weighted mixture and routes that sample to one teacher with weight 1.0.

MOPD_TEACHER_SERVERS="reasoning=http://teacher-reasoning:8000,"\
"codegen=http://teacher-codegen:8000,"\
"agent=http://teacher-agent:8000"
MOPD_TEACHER_HEADS="reasoning=/path/to/reasoning_teacher,"\
"codegen=/path/to/qwen3_6_27b,"\
"agent=/path/to/agent_teacher"
MOPD_TEACHER_WEIGHTS="reasoning=reasoning:0.75,codegen:0.05,agent:0.20;"\
"codegen=codegen:0.75,reasoning:0.05,agent:0.20;"\
"agent=agent:0.75,reasoning:0.05,codegen:0.20"

CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-0.5B \
    --dataset /path/to/mopd_train.jsonl \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --mopd_enable true \
    --mopd_teacher_servers "$MOPD_TEACHER_SERVERS" \
    --mopd_teacher_heads "$MOPD_TEACHER_HEADS" \
    --mopd_teacher_weights "$MOPD_TEACHER_WEIGHTS" \
    --mopd_task_column task \
    --mopd_default_task codegen \
    --mopd_teacher_id_column teacher_id \
    --mopd_hidden_dtype bf16 \
    --mopd_loss_chunk_size 512 \
    --beta 1.0 \
    --sft_alpha 0.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --save_steps 100 \
    --output_dir output/mopd
