#!/bin/bash

# Configuration for Kaggle Environment
# Assuming dataset is mounted at: /kaggle/input/datasets/lcngtr/caer-s
# Assuming this repo is cloned to: /kaggle/working/RAPT-CLIP-CAER
# Assuming annotations are uploaded/generated at: /kaggle/working/RAPT-CLIP-CAER/caer_s_annotations

DATASET_ROOT="/kaggle/input/datasets/lcngtr/caer-s"
ANN_DIR="./caer_s_annotations"

# Placeholder for model path - CHANGE THIS to your trained model checkpoint path! 
MODEL_PATH="outputs/CAER_S_KAGGLE/model_best.pth" 

echo "Starting CAER-S Evaluation on Kaggle..."
echo "Dataset Root: $DATASET_ROOT"
echo "Model Checkpoint: $MODEL_PATH"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model checkpoint not found at $MODEL_PATH"
  echo "Please update MODEL_PATH in this script to point to your .pth file."
  exit 1
fi

python main.py \
  --mode eval \
  --exper-name CAER_S_EVAL_KAGGLE \
  --dataset CAER-S \
  --gpu 0 \
  --batch-size 32 \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --root-dir "$DATASET_ROOT" \
  --test-annotation "$ANN_DIR/test.txt" \
  --bounding-box-face "$ANN_DIR/caer_s_faces.json" \
  --eval-checkpoint "$MODEL_PATH" \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --crop-body

echo "Evaluation Finished!"
