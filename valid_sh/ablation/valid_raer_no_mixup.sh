#!/bin/bash
# VALIDATION: RAER NO MIXUP
# NOTE: Update --eval-checkpoint with your actual trained model path!

python main.py \
  --mode eval \
  --exper-name Eval-RAER-NoMixup \
  --dataset RAER \
  --gpu 0 \
  --batch-size 8 \
  --workers 4 \
  --image-size 224 \
  --root-dir ./dataset/RAER \
  --test-annotation ./dataset/RAER/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face ./dataset/RAER/bounding_box/face.json \
  --bounding-box-body ./dataset/RAER/bounding_box/body.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --crop-body \
  --eval-checkpoint outputs/Ablation-RAER-NoMixup-TIMESTAMP/model_best.pth
