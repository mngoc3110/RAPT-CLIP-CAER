#!/bin/bash
# ABLATION: NO PROMPT TUNING (Static Hand-crafted Prompts)
# Also disables MI/DC losses as they require learnable prompts.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Ablation-RAER-NoPromptTuning \
  --dataset RAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 0.0 \
  --lr-adapter 1e-4 \
  --weight-decay 0.1 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir ./dataset/RAER \
  --train-annotation ./dataset/RAER/annotation/train_80.txt \
  --val-annotation ./dataset/RAER/annotation/val_20.txt \
  --test-annotation ./dataset/RAER/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face ./dataset/RAER/bounding_box/face.json \
  --bounding-box-body ./dataset/RAER/bounding_box/body.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --lambda_dc 0.0 \
  --lambda_mi 0.0 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
