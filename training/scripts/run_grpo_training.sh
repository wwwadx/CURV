#!/bin/bash

# GRPO Training Script for Medical Vision-Language Models
# Adapted from ms-swift GRPO training for CURV repository
#
# Usage: bash run_grpo_training.sh [config_file]
# 
# Requirements:
# - Two GPUs are left for vLLM inference acceleration
# - pip install math_verify # reward function
# - pip install -U trl
# - GPU memory: 8 * 60GiB

# Set default configuration file
CONFIG_FILE=${1:-"../config/grpo_config.py"}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Usage: bash run_grpo_training.sh [config_file]"
    exit 1
fi

# Load configuration (this would be handled by the Python script)
echo "Loading configuration from: $CONFIG_FILE"

# Set environment variables
export MAX_PIXELS=602112
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export NCCL_TIMEOUT=360000

# Original ms-swift command (preserved for reference)
# This command structure is maintained in the Python training script
echo "Starting GRPO training with the following configuration:"
echo "Model: /save/Qwen2.5-VL/qwen-vl-finetune/cxr_reasoning_initial/v0-20250429-011631/checkpoint-14"
echo "Dataset: /save/cxr_report/reasoning_inital_sft_data/qwen_training_data_for_grpo_standard_messages_uncertain.jsonl"
echo "Reward functions: external_cxr_format external_cxr_f1chexbert external_cxr_radgraph_f1 external_cxr_radgraph_uncertain_hard external_cxr_radgraph_uncertain_soft"

# Run the Python training script
python train_grpo.py \
    --config_file "$CONFIG_FILE" \
    --model_id_or_path "/save/Qwen2.5-VL/qwen-vl-finetune/cxr_reasoning_initial/v0-20250429-011631/checkpoint-14" \
    --dataset_path "/save/cxr_report/reasoning_inital_sft_data/qwen_training_data_for_grpo_standard_messages_uncertain.jsonl" \
    --reward_functions "format_cxr,f1_chexbert_cxr,radgraph_f1_cxr" \
    --use_vllm true \
    --vllm_server_host "127.0.0.1" \
    --vllm_server_port 8000 \
    --train_type "full" \
    --torch_dtype "bfloat16" \
    --max_completion_length 1536 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-7 \
    --do_eval false \
    --eval_steps 100000000000 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --output_dir "output" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_grad_norm 0.5 \
    --dataset_num_proc 4 \
    --split_dataset_ratio 0 \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate false \
    --system_prompt_file "../prompts/prompt_cxr.txt" \
    --deepspeed "zero3" \
    --ddp_timeout 180000000 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1 \
    --report_to "tensorboard"

# Original ms-swift command for reference:
# MAX_PIXELS=602112 \
# CUDA_VISIBLE_DEVICES=0,1 \
# NPROC_PER_NODE=2 \
# NCCL_TIMEOUT=360000 \
# swift rlhf \
#     --rlhf_type grpo \
#     --model /save/Qwen2.5-VL/qwen-vl-finetune/cxr_reasoning_initial/v0-20250429-011631/checkpoint-14 \
#     --model_kwargs '{"weights_only": false}' \
#     --external_plugins /save/ms-swift/examples/train/grpo/plugin/plugin.py \
#     --reward_funcs external_cxr_format external_cxr_f1chexbert external_cxr_radgraph_f1 external_cxr_radgraph_uncertain_hard external_cxr_radgraph_uncertain_soft \
#     --use_vllm true \
#     --vllm_server_host 127.0.0.1 \
#     --vllm_server_port 8000 \
#     --vllm_device auto \
#     --train_type full \
#     --torch_dtype bfloat16 \
#     --dataset /save/cxr_report/reasoning_inital_sft_data/qwen_training_data_for_grpo_standard_messages_uncertain.jsonl \
#     --max_completion_length 1536 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-7 \
#     --do_eval False \
#     --eval_steps 100000000000 \
#     --save_steps 1000 \
#     --save_total_limit 100 \
#     --logging_steps 1 \
#     --output_dir output \
#     --warmup_ratio 0.05 \
#     --dataloader_num_workers 4 \
#     --max_grad_norm 0.5 \
#     --dataset_num_proc 4 \
#     --split_dataset_ratio 0 \
#     --num_generations 8 \
#     --temperature 1.0 \
#     --top_p 0.9 \
#     --top_k 50 \
#     --async_generate false \
#     --system '/save/ms-swift/examples/train/grpo/prompt_cxr.txt' \
#     --deepspeed zero3 \
#     --ddp_timeout 180000000 \
#     --log_completions true \
#     --num_iterations 1 \
#     --num_infer_workers 1 \
#     --report_to tensorboard