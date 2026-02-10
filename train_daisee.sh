#!/bin/bash

# =================================================================
# DAiSEE ENGAGEMENT PIPELINE (KAGGLE VERSION)
# Strategy: 
# 1. Use existing daisee_*.txt files from Kaggle input.
# 2. Save fixed annotations to a writable local directory.
# 3. Start training for 4 engagement levels.
# =================================================================

# Define Data Directories (Kaggle Input)
ROOT_DATA_DIR="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
# Writable directory for annotations on Kaggle
ANNOT_DIR="./daisee_annotations"

# Ensure annotation directory exists
mkdir -p "$ANNOT_DIR"

# --- Step 1: Fix Paths in Annotation Files ---
echo "=> Preparing DAiSEE Annotations for Kaggle..."

python3 - <<EOF
import os

# Use the ROOT_DATA_DIR defined in bash
source_dir = "$ROOT_DATA_DIR"
target_dir = "$ANNOT_DIR"

files = {
    "daisee_train.txt": "train.txt",
    "daisee_val.txt": "val.txt",
    "daisee_test.txt": "test.txt"
}

for src, dst in files.items():
    src_path = os.path.join(source_dir, src)
    dst_path = os.path.join(target_dir, dst)
    
    if not os.path.exists(src_path):
        print(f"Warning: Source {src_path} not found.")
        continue
        
    with open(src_path, 'r') as f:
        lines = f.readlines()
        
    # On Kaggle, we keep the paths as they are since root-dir will be /kaggle/input/datasets/mngochocsupham/daisee/
    # And lines already start with DAiSEE_data/
    fixed_lines = lines 
        
    with open(dst_path, 'w') as f:
        f.writelines(fixed_lines)
    print(f"Copied {len(fixed_lines)} lines for {dst}")

print("Done!")
EOF

if [ $? -ne 0 ]; then
    echo "Error: Python script failed."
    exit 1
fi

# --- Step 2: Training ---
echo "=> Starting Training on Kaggle..."

# DAiSEE videos are typically 10 seconds, sampled at 30fps = 300 frames.
# We'll sample 16 segments from each video.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-Kaggle \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  --root-dir /kaggle/input/datasets/mngochocsupham/daisee/ \
  --train-annotation "$ANNOT_DIR/train.txt" \
  --val-annotation "$ANNOT_DIR/val.txt" \
  --test-annotation "$ANNOT_DIR/test.txt" \
  --clip-path ViT-B/16 \
  --bounding-box-face "$ANNOT_DIR/dummy_box.json" \
  --bounding-box-body "$ANNOT_DIR/dummy_box.json" \
  --text-type prompt_ensemble \
  --lambda_dc 0.1 \
  --lambda_mi 0.1 \
  --temperature 0.07 \
  --use-ldl \
  --use-moco \
  --moco-k 2048 \
  --moco-m 0.999 \
  --moco-t 0.07 \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
