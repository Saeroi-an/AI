#!/bin/bash

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)


MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/home/jwlee/volume/Qwen2-vl-finetune-wo/output/checkpoints"
CACHE_DIR="/home/jwlee/volume/Qwen2-vl-finetune-wo/output/cache"  


TRAIN_DATA_FILE="/home/jwlee/volume/Qwen2-vl-finetune-wo/data/ko_zh_datasets_4/train_zh_ko.json" 
VAL_DATA_PATH="/home/jwlee/volume/Qwen2-vl-finetune-wo/data/ko_zh_datasets_4/val_zh_ko.json" 


torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /home/jwlee/volume/Qwen2-vl-finetune-wo/src/train/train_qwen.py \
    --model_name_or_path $MODEL_PATH \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --dataset_use $TRAIN_DATA_FILE \
    --val_data_path $VAL_DATA_PATH \
    --bf16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 4096 \
    --data_flatten True \
    --data_packing True \
    --max_pixels 2000000 \
    --min_pixels 200000 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --deepspeed zero3.json \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --load_best_model_at_end True \
    --logging_steps 10 \
    --report_to "tensorboard" \

    
    