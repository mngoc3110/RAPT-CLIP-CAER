#!/bin/bash

# =========================================================================================
# CAER VIDEO TRAINING SCRIPT (Targeting >90% Accuracy)
# Strategy: Use 16 frames (Temporal Information), Face + Context Dual Stream
# =========================================================================================

# Adjust this to where the dataset is located on Kaggle
DATASET_ROOT="/kaggle/input/caer-data/CAER"
ANN_DIR="./caer_video_annotations"

echo "Starting CAER Video Training on Kaggle..."
echo "Dataset Root: $DATASET_ROOT"
echo "Annotations: $ANN_DIR"

python main.py \
  --mode train \
  --exper-name CAER_VIDEO_FULL \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 16 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$ANN_DIR/train.txt" \
  --val-annotation "$ANN_DIR/validation.txt" \
  --test-annotation "$ANN_DIR/test.txt" \
  --bounding-box-face "$ANN_DIR/caer_video_faces.json" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --ldam-max-m 0.3 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --label-smoothing 0.05 \
  --use-amp \
  --grad-clip 1.0

echo "Training Finished!"
