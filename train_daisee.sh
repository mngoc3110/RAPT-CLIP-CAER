#!/bin/bash

# =================================================================
# DAiSEE ENGAGEMENT PIPELINE (KAGGLE VERSION)
# Strategy: 
# 1. Use existing daisee_*.txt files from Kaggle input.
# 2. Smartly map IDs to actual video files or frame folders.
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
import glob

# Use the ROOT_DATA_DIR defined in bash
source_dir = "$ROOT_DATA_DIR"
target_dir = "$ANNOT_DIR"
root_kaggle_dir = "/kaggle/input/datasets/mngochocsupham/daisee/"

files = {
    "daisee_train.txt": "train.txt",
    "daisee_val.txt": "val.txt",
    "daisee_test.txt": "test.txt"
}

# Helper to find the actual path of a clip
def find_clip_path(clip_id, split_folder):
    # Search for video file with clip_id
    # Pattern 1: .../Train/ClipID.avi
    # Pattern 2: .../Train/SubjectID/ClipID.avi
    # We search recursively in the split folder (Train/Test/Validation)
    
    search_path = os.path.join(root_kaggle_dir, split_folder)
    
    # Try finding video file
    video_files = glob.glob(os.path.join(search_path, "**", f"{clip_id}.*"), recursive=True)
    video_files = [f for f in video_files if f.endswith(('.avi', '.mp4', '.mov'))]
    
    if video_files:
        # Return path relative to root_kaggle_dir for dataloader
        rel_path = os.path.relpath(video_files[0], root_kaggle_dir)
        return rel_path
        
    # Try finding frame folder
    frame_folders = glob.glob(os.path.join(search_path, "**", clip_id, "frames"), recursive=True)
    if frame_folders:
        rel_path = os.path.relpath(frame_folders[0], root_kaggle_dir)
        return rel_path
        
    return None

for src, dst in files.items():
    src_path = os.path.join(source_dir, src)
    dst_path = os.path.join(target_dir, dst)
    
    if not os.path.exists(src_path):
        print(f"Warning: Source {src_path} not found.")
        continue
    
    # Determine split folder name from filename
    if "train" in src: split_folder = "Train"
    elif "val" in src: split_folder = "Validation"
    elif "test" in src: split_folder = "Test"
    else: split_folder = ""
        
    with open(src_path, 'r') as f:
        lines = f.readlines()
        
    fixed_lines = []
    found_count = 0
    missing_count = 0
    
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 3: continue
        
        orig_path = parts[0] # e.g., DAiSEE_data/DataSet/Train/110001/1100011002/frames
        
        # Extract Clip ID. Usually the second to last part if frames is last.
        # 1100011002/frames -> 1100011002
        # Or just filename without extension if it's a video path
        
        path_parts = orig_path.split('/')
        if path_parts[-1] == 'frames':
            clip_id = path_parts[-2]
        else:
            clip_id = os.path.splitext(path_parts[-1])[0]
            
        actual_path = find_clip_path(clip_id, split_folder)
        
        if actual_path:
            # Reconstruct line with new path
            # Keep num_frames and label from original line
            new_line = f"{actual_path} {parts[1]} {parts[2]}\n"
            fixed_lines.append(new_line)
            found_count += 1
        else:
            # print(f"Missing: {clip_id}")
            missing_count += 1
        
    with open(dst_path, 'w') as f:
        f.writelines(fixed_lines)
    print(f"Processed {dst}: Found {found_count}, Missing {missing_count}")

# Create dummy box file in the writable annotation directory
import json
dummy_box_path = os.path.join(target_dir, "dummy_box.json")
with open(dummy_box_path, "w") as f:
    json.dump({}, f)
print(f"Created dummy box file at: {dummy_box_path}")

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
