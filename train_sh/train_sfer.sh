#!/bin/bash

# =================================================================
# AUTOMATED SFER PIPELINE
# Strategy: 
# 1. Use existing Train/Test folders.
# 2. Map classes to IDs 1-7.
# 3. Generate annotation files.
# =================================================================

# Define Data Directories
ROOT_DATA_DIR="./dataset/Student_FER_Dataset"
# Path to the actual image folders
TRAIN_DIR="$ROOT_DATA_DIR/raw/SFER dataset/SFER dataset/train"
TEST_DIR="$ROOT_DATA_DIR/raw/SFER dataset/SFER dataset/test"
ANNOT_DIR="$ROOT_DATA_DIR/annotations"

# Ensure annotation directory exists
mkdir -p "$ANNOT_DIR"

# --- Step 1: Generate Annotation Files ---
echo "=> Automatically Generating SFER Annotations..."

python3 - <<EOF
import os
import glob
import json

TRAIN_DIR = "$TRAIN_DIR"
TEST_DIR = "$TEST_DIR"
ANNOTATION_DIR = "$ANNOT_DIR"

# Classes in alphabetical order to match model definition
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
class_to_idx = {cls_name: i+1 for i, cls_name in enumerate(classes)}

def generate_list(source_dir, output_file):
    lines = []
    if not os.path.exists(source_dir):
        print(f"Error: Directory {source_dir} not found.")
        return
    
    for cls_name in classes:
        cls_dir = os.path.join(source_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"Warning: Class directory {cls_dir} not found.")
            continue
            
        # Search for images (jpg, png, jpeg)
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            images.extend(glob.glob(os.path.join(cls_dir, ext)))
        
        for img_path in images:
            # We need relative path from where the dataloader runs (usually project root)
            # Or absolute path. Let's use relative to project root if possible, or keep absolute.
            # The CK+ script uses paths relative to ROOT_DATA_DIR in the Python script?
            # No, CK+ script generates paths like "CKPlus_Dataset/raw/..." which are relative to project root.
            # So we should use paths relative to project root if we run main.py from project root.
            
            # Since existing paths are absolute or relative to where we are, let's just use what we have found
            # but ensure they are correct relative to where main.py is run.
            
            # glob returns paths as provided in input pattern.
            # If TRAIN_DIR is relative to project root, then img_path is too.
            lines.append(f"{img_path} 1 {class_to_idx[cls_name]}\n")
            
    with open(output_file, "w") as f:
        f.writelines(lines)
    print(f"Generated {len(lines)} samples for {output_file}")

generate_list(TRAIN_DIR, os.path.join(ANNOTATION_DIR, "train.txt"))
generate_list(TEST_DIR, os.path.join(ANNOTATION_DIR, "test.txt"))

# Use test set as validation set for now
import shutil
shutil.copy(os.path.join(ANNOTATION_DIR, "test.txt"), os.path.join(ANNOTATION_DIR, "val.txt"))

# Create dummy box file
with open(os.path.join(ANNOTATION_DIR, "dummy_box.json"), "w") as f:
    json.dump({}, f)

print("Done!")
EOF

if [ $? -ne 0 ]; then
    echo "Error: Python script failed."
    exit 1
fi

# --- Step 2: Training ---
echo "=> Starting Training..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python main.py \
  --mode train \
  --exper-name Train-SFER-Stable \
  --dataset SFER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 32 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 2e-5 \
  --lr-adapter 2e-5 \
  --weight-decay 0.01 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 5 \
  --root-dir ./ \
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
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2