#!/bin/bash

# Configuration for Kaggle Environment
DATASET_ROOT="/kaggle/input/datasets/lcngtr/caer-s"
ANN_DIR="./caer_s_annotations"

# Placeholder for model path - CHANGE THIS to your trained model checkpoint path!
MODEL_PATH="outputs/CAER_S_KAGGLE/model_best.pth"

echo "Starting CAER-S TTA (FiveCrop) Evaluation on Kaggle..."
echo "Dataset Root: $DATASET_ROOT"
echo "Model Checkpoint: $MODEL_PATH"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model checkpoint not found at $MODEL_PATH"
  echo "Please update MODEL_PATH in this script to point to your .pth file."
  exit 1
fi

python evaluate_tta.py \
  --dataset CAER-S \
  --root-dir "$DATASET_ROOT" \
  --test-annotation "$ANN_DIR/test.txt" \
  --bounding-box-face "$ANN_DIR/caer_s_faces.json" \
  --checkpoint "$MODEL_PATH" \
  --batch-size 8 \
  --image-size 224 \
  --gpu 0 \
  --temporal-layers 1 \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --use-moco

echo "TTA Evaluation Finished!"
