#!/bin/bash

# Training script for CAER-S (Static Image) dataset - Optimized for MacBook Pro

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train_CAER_S_Optimized \
  --dataset CAER-S \
  --gpu mps \
  --epochs 20 \
  --batch-size 4 \
  --accumulation-steps 8 \
  --optimizer AdamW \
  --lr 5e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-3 \
  --lr-adapter 5e-4 \
  --weight-decay 0.0005 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir CAER-S \
  --clip-path ViT-B/16 \
  --text-type prompt_ensemble \
  --temporal-type attn_pool \
  --use-adapter True \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --lambda_dc 0.05 \
  --dc-warmup 3 \
  --dc-ramp 10 \
  --lambda_mi 0.05 \
  --mi-warmup 3 \
  --mi-ramp 10 \
  --temperature 0.07 \
  --use-ldl \
  --ldl-temperature 1.0 \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2 \
  --use-weighted-sampler