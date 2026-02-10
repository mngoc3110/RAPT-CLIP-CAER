#!/bin/bash

# =================================================================
# AUTOMATED CK+ PIPELINE (STABLE IMAGE-BASED SPLIT)
# Strategy: 
# 1. Each image is a sample.
# 2. Split by Sequence ID (70% Train, 15% Val, 15% Test).
# 3. Stable Split using seed 42.
# =================================================================

# Define Data Directories
ROOT_DATA_DIR="./CKPlus_Dataset"
RAW_DIR="$ROOT_DATA_DIR/raw/CK+48"
ANNOT_DIR="$ROOT_DATA_DIR/annotations"

# Ensure root directory exists
mkdir -p "$ANNOT_DIR"

# --- Step 1: Split Data ---
echo "=> Automatically Splitting CK+ Data (Stable Split)..."

python3 - <<EOF
import os
import glob
import random
import json

SOURCE_DIR = "$RAW_DIR"
ANNOTATION_DIR = "$ANNOT_DIR"
# Fixed seed for stable split
random.seed(42)

if not os.path.exists(SOURCE_DIR):
    print(f"Error: Source directory {SOURCE_DIR} not found.")
    exit(1)

classes = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])
class_to_idx = {cls_name: i+1 for i, cls_name in enumerate(classes)}
train_lines, val_lines, test_lines = [], [], []
sequences = {}

for cls_name in classes:
    for img_path in glob.glob(os.path.join(SOURCE_DIR, cls_name, "*.png")):
        filename = os.path.basename(img_path)
        # Fix: Ensure filename split works correctly
        parts = filename.split("_")
        if len(parts) >= 2:
            seq_id = f"{parts[0]}_{parts[1]}"
        else:
            seq_id = filename.split(".")[0] # Fallback
            
        if seq_id not in sequences:
            sequences[seq_id] = {"cls": cls_name, "images": []}
        sequences[seq_id]["images"].append(img_path)

seq_ids = list(sequences.keys())
random.shuffle(seq_ids)

n_total = len(seq_ids)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)

train_ids = seq_ids[:n_train]
val_ids = seq_ids[n_train:n_train+n_val]
test_ids = seq_ids[n_train+n_val:]

for sid in seq_ids:
    for img in sequences[sid]["images"]:
        line = f"{img} 1 {class_to_idx[sequences[sid]['cls']]}\n"
        if sid in train_ids:
            train_lines.append(line)
        elif sid in val_ids:
            val_lines.append(line)
        else:
            test_lines.append(line)

with open(os.path.join(ANNOTATION_DIR, "train.txt"), "w") as f:
    f.writelines(train_lines)
with open(os.path.join(ANNOTATION_DIR, "val.txt"), "w") as f:
    f.writelines(val_lines)
with open(os.path.join(ANNOTATION_DIR, "test.txt"), "w") as f:
    f.writelines(test_lines)
with open(os.path.join(ANNOTATION_DIR, "dummy_box.json"), "w") as f:
    json.dump({}, f)

print(f"Done! Total: {len(train_lines)+len(val_lines)+len(test_lines)} images.")
print(f"Train: {len(train_lines)} | Val: {len(val_lines)} | Test: {len(test_lines)}")
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
  --exper-name Train-CKPlus-Flat \
  --dataset CK+ \
  --gpu 0 \
  --epochs 50 \
  --batch-size 32 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 30 45 \
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
  --use-ldl \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
