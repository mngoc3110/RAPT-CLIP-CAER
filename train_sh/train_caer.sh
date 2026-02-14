#!/bin/bash

# =================================================================
# CAER VIDEO PIPELINE (FULL DUAL-STREAM)
# =================================================================

# Define Directories
CAER_ROOT="./dataset/CAER"
ANNOT_DIR="./caer_annotations"
mkdir -p "$ANNOT_DIR"

echo "=> Generating CAER Annotations..."

python3 - <<EOF
import os
import glob
import json
import random

CAER_ROOT = "$CAER_ROOT"
ANNOT_DIR = "$ANNOT_DIR"

# CAER Classes
classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
class_to_idx = {cls_name: i+1 for i, cls_name in enumerate(classes)}

def get_video_list(split_name):
    lines = []
    split_dir = os.path.join(CAER_ROOT, split_name)
    
    if not os.path.exists(split_dir):
        print(f"Warning: {split_dir} not found.")
        return []

    for cls_name in classes:
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        videos = glob.glob(os.path.join(cls_dir, "*.avi"))
        for v_path in videos:
            lines.append(f"{v_path} 100 {class_to_idx[cls_name]}\n")
    return lines

# 1. Get all training samples (include original 'validation' folder if exists to boost training)
train_samples = get_video_list("train")
val_folder_samples = get_video_list("validation") 
train_samples.extend(val_folder_samples) # Merge them

# 2. Shuffle and Split 80/20
random.seed(42)
random.shuffle(train_samples)

split_idx = int(len(train_samples) * 0.8)
train_lines = train_samples[:split_idx]
val_lines = train_samples[split_idx:]

print(f"Total training pool: {len(train_samples)}")
print(f"-> Train set: {len(train_lines)} (80%)")
print(f"-> Val set:   {len(val_lines)} (20%)")

with open(os.path.join(ANNOT_DIR, "train.txt"), "w") as f:
    f.writelines(train_lines)
with open(os.path.join(ANNOT_DIR, "val.txt"), "w") as f:
    f.writelines(val_lines)

# 3. Generate Test set (keep original test set intact)
test_lines = get_video_list("test")
with open(os.path.join(ANNOT_DIR, "test.txt"), "w") as f:
    f.writelines(test_lines)
print(f"-> Test set:  {len(test_lines)} (Original Test)")

EOF

echo "=> Starting Training (CAER Dual-Stream)..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CAER setup: 
# - Use real bounding boxes for Face/Body
# - num-segments 8 or 16 for temporal learning
# - dataset RAER (vì dataloader RAER hỗ trợ dual-stream và bbox tốt nhất)
# - Lưu ý: Tôi đổi --dataset thành RAER để tận dụng video_dataloader.py 
#   nhưng trỏ vào data CAER.

python main.py \
  --mode train \
  --exper-name Train-CAER-DualStream-Tuned \
  --dataset CAER \
  --gpu 0 \
  --epochs 30 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-3 \
  --lr-adapter 1e-4 \
  --weight-decay 0.2 \
  --milestones 10 20 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 4 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 20 \
  --root-dir "./" \
  --train-annotation "$ANNOT_DIR/train.txt" \
  --val-annotation "$ANNOT_DIR/val.txt" \
  --test-annotation "$ANNOT_DIR/test.txt" \
  --clip-path ViT-B/16 \
  --bounding-box-face "$CAER_ROOT/bounding_box/face.json" \
  --bounding-box-body "$CAER_ROOT/bounding_box/body.json" \
  --text-type prompt_ensemble \
  --lambda_dc 0.05 \
  --lambda_mi 0.05 \
  --use-amp \
  --grad-clip 1.0 \
  --use-weighted-sampler \
  --mixup-alpha 0.2