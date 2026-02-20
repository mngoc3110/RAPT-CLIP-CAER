#!/bin/bash

# =========================================================================================
# CAER VIDEO TRAINING SCRIPT (Transfer Learning from RAER)
# =========================================================================================

# 1. Dataset Path
DATASET_ROOT="/kaggle/input/datasets/harinath07/caer-data/CAER"
ANN_DIR="$(pwd)/caer_video_annotations"

# 2. Transfer Learning: Use Best RAER Model (Found in Kaggle Input)
RESUME_PATH="/kaggle/input/datasets/mngochocsupham/model-path/Ablation-RAER-FULL_fix_parameter_fullface_train_test-[02-15]-[07:38]/model_best.pth"

echo "Checking annotation directory: $ANN_DIR"
ls -lh "$ANN_DIR"

echo "Starting CAER Video Training on Kaggle..."
echo "Using Transfer Learning from RAER: $RESUME_PATH"

# Logic to handle resume/transfer
if [ -n "$RESUME_PATH" ] && [ -f "$RESUME_PATH" ]; then
  RESUME_ARG="--resume $RESUME_PATH"
else
  echo "Warning: RAER model not found at $RESUME_PATH. Training from CLIP baseline."
  RESUME_ARG=""
fi

# 3. Aggressive Training Parameters for >74% Accuracy
python main.py \
  --mode train \
  --exper-name CAER_VIDEO_TRANSFER_RAER \
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
  --contexts-number 16 \
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
