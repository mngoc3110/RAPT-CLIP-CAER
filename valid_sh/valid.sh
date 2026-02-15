#!/bin/bash

# NOTE: Update --eval-checkpoint with the actual path to your trained model
# e.g., outputs/Train-[date]-[time]/model_best.pth

python main.py \
    --mode eval \
    --gpu 0 \
    --exper-name Eval \
    --eval-checkpoint outputs/Train/model_best.pth \
    --train-annotation RAER/annotation/train_80.txt \
    --val-annotation RAER/annotation/val_20.txt \
    --test-annotation RAER/annotation/test.txt \
    --clip-path ViT-B/16 \
    --bounding-box-face RAER/bounding_box/face.json \
    --bounding-box-body RAER/bounding_box/body.json \
    --text-type prompt_ensemble \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --temperature 0.07 \
    --crop-body