#!/bin/bash

# Training script for CAER (Video) dataset

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train_CAER_Video \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.0005 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  --root-dir /kaggle/input/caer-video-dataset/CAER \
  --clip-path ViT-B/16 \
  --text-type prompt_ensemble \
  --temporal-type attn_pool \
  --use-adapter True \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --lambda_dc 0.1 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --temperature 0.07 \
  --use-ldl \
  --ldl-temperature 1.0 \
  --use-amp \
  --grad-clip 1.0 \


