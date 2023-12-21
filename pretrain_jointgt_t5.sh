#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python /content/JointGT_VN/cli_gt.py \
        --do_train \
        --do_pretrain \
        --model_name t5 \
        --model_path VietAI/vit5-base \
        --tokenizer_path VietAI/vit5-base \
        --output_dir pretrain_model/jointgt_t5 \
        --train_file /content/drive/MyDrive/jointgt_data/w_train \
        --predict_file /content/drive/MyDrive/jointgt_data/w_test \
        --knowledge_file /content/drive/MyDrive/jointgt_data/knowledge-full \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --max_input_length 2000 \
        --max_output_length 64 \
        --gradient_accumulation_steps 8 \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --warmup_steps 2700 \
        --eval_period 3000 \
        --save_period 40000 \
        --num_workers 4
