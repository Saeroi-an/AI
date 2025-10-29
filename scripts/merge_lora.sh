#!/bin/bash

# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

# GPU 환경 설정 (train env, GPU 2~3)
export CUDA_VISIBLE_DEVICES=0,1

python /home/jwlee/volume/Qwen2-vl-finetune-wo/src/merge_lora_weights.py \
    --model-path /home/jwlee/volume/Qwen2-vl-finetune-wo/output/zh_ko_qwen2vl
    --model-base $MODEL_NAME  \
    --save-model-path /home/jwlee/volume/Qwen2-vl-finetune-wo/output/merge_zh_ko
    --safe-serialization