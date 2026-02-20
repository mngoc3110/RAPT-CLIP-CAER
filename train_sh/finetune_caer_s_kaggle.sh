#!/bin/bash

# =========================================================================================
# CAER-S FINE-TUNING SCRIPT (Boosting Accuracy)
# Strategy: Lower LR, Higher Regularization, Resume from Best Checkpoint
# =========================================================================================

DATASET_ROOT="/kaggle/input/datasets/lcngtr/caer-s"
ANN_DIR="./caer_s_annotations"

# IMPORTANT: Point this to your BEST checkpoint from the previous training run
PRETRAINED_PATH="/kaggle/input/datasets/mngochocsupham/model-path/Ablation-RAER-FULL_fix_parameter_fullface_train_test-[02-15]-[07:38]/model_best.pth"

echo "Starting CAER-S Fine-tuning on Kaggle..."
echo "Resuming from: $PRETRAINED_PATH"

if [ ! -f "$PRETRAINED_PATH" ]; then
  echo "⚠️ WARNING: Checkpoint not found at $PRETRAINED_PATH"
  echo "Please edit PRETRAINED_PATH in this script before running!"
  # exit 1 # Uncomment to enforce check
fi

python main.py \
  --mode train \
  --exper-name CAER_S_FINETUNE \
  --dataset CAER-S \
  --gpu 0 \
  --epochs 10 \
  --batch-size 32 \
  --resume "$PRETRAINED_PATH" \
  --optimizer AdamW \
  --lr 1e-6 \
  --lr-image-encoder 1e-7 \
  --lr-prompt-learner 2e-5 \
  --lr-adapter 1e-5 \
  --weight-decay 0.1 \
  --temporal-layers 1 \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$DATASET_ROOT" \
  --train-annotation "$ANN_DIR/train.txt" \
  --val-annotation "$ANN_DIR/test.txt" \
  --test-annotation "$ANN_DIR/test.txt" \
  --bounding-box-face "$ANN_DIR/caer_s_faces.json" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 0 \
  --mi-ramp 1 \
  --dc-warmup 0 \
  --dc-ramp 1 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --use-amp \
  --crop-body \
  --grad-clip 1.0

echo "Fine-tuning Finished!"
