#!/bin/bash

# =========================================================================================
# CAER VIDEO TRAINING SCRIPT (Targeting >90% Accuracy)
# Strategy: Use 16 frames (Temporal Information), Face + Context Dual Stream
# =========================================================================================

# Adjust this to where the dataset is located on Kaggle
DATASET_ROOT="/kaggle/input/datasets/harinath07/caer-data/CAER"
# Use absolute path for annotations to avoid FileNotFoundError
ANN_DIR="$(pwd)/caer_video_annotations"

# [OPTIONAL] Resume from a checkpoint (e.g., from Epoch 5)
# Set this to the path of your checkpoint file on Kaggle.
# Leave empty ("") to start from scratch.
RESUME_PATH="/kaggle/input/datasets/mngochocsupham/model-caer/model_caer_video.pth"

echo "Checking annotation directory: $ANN_DIR"
ls -lh "$ANN_DIR"

echo "Starting CAER Video Training on Kaggle..."
echo "Dataset Root: $DATASET_ROOT"
echo "Annotations: $ANN_DIR"

if [ -n "$RESUME_PATH" ] && [ -f "$RESUME_PATH" ]; then
  echo "Resuming from: $RESUME_PATH"
  RESUME_ARG="--resume $RESUME_PATH"
else
  echo "No valid checkpoint found at $RESUME_PATH, starting from scratch."
  RESUME_ARG=""
fi

python main.py \
  --mode train \
  --exper-name CAER_VIDEO_FULL \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.1 \
  --temporal-layers 1 \
  --num-segments 8 \
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
  --ldam-max-m 0.5 \
  --ldam-s 50.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --label-smoothing 0.1 \
  --use-amp \
  --grad-clip 1.0 \
  $RESUME_ARG

echo "Training Finished!"
