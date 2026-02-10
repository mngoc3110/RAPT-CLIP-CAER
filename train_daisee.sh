#!/bin/bash

# =================================================================
# DAiSEE ENGAGEMENT PIPELINE (KAGGLE STABLE VERSION)
# =================================================================

# Define Data Directories (Kaggle Input)
# Dựa trên đường dẫn bạn cung cấp: /kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels/TestLabels.csv
KAGGLE_DATASET_ROOT="/kaggle/input/datasets/mngochocsupham/daisee"
ROOT_DATA_DIR="$KAGGLE_DATASET_ROOT/DAiSEE_data"
# Writable directory for annotations on Kaggle
ANNOT_DIR="./daisee_annotations"

# Ensure annotation directory exists
mkdir -p "$ANNOT_DIR"

# --- Step 1: Fix Paths in Annotation Files ---
echo "=> Preparing DAiSEE Annotations for Kaggle..."

python3 - <<EOF
import os
import glob
import json

source_dir = "$ROOT_DATA_DIR"
target_dir = "$ANNOT_DIR"
root_kaggle_dir = "$KAGGLE_DATASET_ROOT"

files = {
    "daisee_train.txt": "train.txt",
    "daisee_val.txt": "val.txt",
    "daisee_test.txt": "test.txt"
}

def find_clip_path(clip_id, split_name):
    # Tìm kiếm file video hoặc folder frames chứa Clip ID
    # split_name: Train, Test, Validation
    
    # Thử tìm trực tiếp trong thư mục DataSet của split đó
    search_pattern = os.path.join(root_kaggle_dir, "**", f"{clip_id}*")
    candidates = glob.glob(search_pattern, recursive=True)
    
    # Ưu tiên folder 'frames' nếu có
    for c in candidates:
        if os.path.isdir(c) and c.endswith("frames"):
            return os.path.relpath(c, root_kaggle_dir)
            
    # Ưu tiên file video (.avi, .mp4)
    for c in candidates:
        if os.path.isfile(c) and c.lower().endswith((".avi", ".mp4", ".mov")):
            return os.path.relpath(c, root_kaggle_dir)
            
    # Thử folder cha của frames
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "frames")):
            return os.path.relpath(os.path.join(c, "frames"), root_kaggle_dir)
            
    return None

for src, dst in files.items():
    # Tìm file txt nguồn (có thể nằm ở ROOT_DATA_DIR hoặc KAGGLE_DATASET_ROOT)
    src_path = os.path.join(source_dir, src)
    if not os.path.exists(src_path):
        src_path = os.path.join(root_kaggle_dir, src)
        
    if not os.path.exists(src_path):
        print(f"Warning: Source {src} not found in {source_dir} or {root_kaggle_dir}")
        continue
        
    dst_path = os.path.join(target_dir, dst)
    
    with open(src_path, 'r') as f:
        lines = f.readlines()
        
    fixed_lines = []
    found = 0
    missing = 0
    
    split_name = "Train" if "train" in src else ("Test" if "test" in src else "Validation")
    
    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) < 3: continue
        
        # Lấy Clip ID từ đường dẫn cũ
        # Ví dụ: DAiSEE_data/DataSet/Train/110001/1100011002/frames -> 1100011002
        orig_path = parts[0]
        path_parts = orig_path.split('/')
        clip_id = path_parts[-2] if path_parts[-1] == 'frames' else os.path.splitext(path_parts[-1])[0]
        
        actual_rel_path = find_clip_path(clip_id, split_name)
        
        if actual_rel_path:
            fixed_lines.append(f"{actual_rel_path} {parts[1]} {parts[2]}\n")
            found += 1
        else:
            missing += 1
            
    with open(dst_path, 'w') as f:
        f.writelines(fixed_lines)
    print(f"Processed {dst}: Found {found}, Missing {missing}")

# Create dummy box file
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
  --root-dir "$KAGGLE_DATASET_ROOT/" \
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