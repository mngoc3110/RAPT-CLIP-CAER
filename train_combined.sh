#!/bin/bash

# Training script for Combined CAER-S (Images) and CAER (Video) dataset

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train_CAER_Combined \
  --dataset CAER_Combined \
  --gpu mps \
  --epochs 30 \
  --batch-size 4 \
  --accumulation-steps 2 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.0005 \
  --milestones 15 25 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir . \
  --clip-path ViT-B/16 \
  --text-type prompt_ensemble \
  --temporal-type attn_pool \
  --use-adapter True \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --temperature 0.07 \
  --use-ldl
