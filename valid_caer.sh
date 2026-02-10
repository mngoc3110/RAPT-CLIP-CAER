#!/bin/bash

# Validation script for CAER (Video) dataset
# Please replace --eval-checkpoint with your actual model path.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode eval \
  --dataset CAER \
  --root-dir CAER \
  --gpu 0 \
  --batch-size 4 \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --clip-path ViT-B/16 \
  --text-type prompt_ensemble \
  --temporal-type attn_pool \
  --use-adapter True \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --eval-checkpoint outputs/Train_CAERS-[02-10]-[21:41]/model_best.pth
