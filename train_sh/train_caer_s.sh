#!/bin/bash
# Training script for CAER-S dataset (Static Images) with Auto-Preprocessing

# Path to CAER-S dataset root. 
# Structure should be:
# ROOT/
#   train/
#     Anger/
#       bbox_train_Angry.csv
#       ...images...
#   test/
#     Anger/
#       bbox_test_Angry.csv
#       ...images...
ROOT_DIR="/Users/macbook/Downloads/CAER-S" 

# Output directory for annotations
ANNOTATION_DIR="CAER_S_annotations"

# Check if dataset exists
if [ ! -d "$ROOT_DIR" ]; then
  echo "Error: Directory $ROOT_DIR does not exist."
  echo "Please edit this script to point to your CAER-S dataset location."
  exit 1
fi

echo "Step 1: Preprocessing CAER-S dataset..."
python utils/preprocess_caer.py --root_dir "$ROOT_DIR" --output_dir "$ANNOTATION_DIR"

echo "Step 2: Starting Training..."
python main.py \
    --dataset CAER-S \
    --root-dir "$ROOT_DIR" \
    --exper-name CAER_S_Optimized \
    --train-annotation "$ANNOTATION_DIR/train.txt" \
    --val-annotation "$ANNOTATION_DIR/test.txt" \
    --test-annotation "$ANNOTATION_DIR/test.txt" \
    --bounding-box-face "$ANNOTATION_DIR/caer_s_faces.json" \
    --batch-size 32 \
    --epochs 50 \
    --lr 5e-5 \
    --lr-image-encoder 5e-6 \
    --lr-prompt-learner 2e-4 \
    --lr-adapter 1e-4 \
    --num-segments 1 \
    --duration 1 \
    --use-weighted-sampler \
    --loss-type ldl \
    --ldl-temperature 1.0 \
    --mixup-alpha 0.4 \
    --gpu mps \
    --workers 4 \
    --print-freq 10 \
    --text-type prompt_ensemble
