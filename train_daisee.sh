#!/bin/bash

# =================================================================
# DAiSEE ENGAGEMENT PIPELINE (KAGGLE CSV-BASED VERSION)
# Strategy: 
# 1. Scan Kaggle directory to find actual paths of all videos.
# 2. Parse DAiSEE CSV labels to get correct Engagement levels.
# 3. Generate clean train/val/test.txt annotations with progress bar.
# =================================================================

# Define Data Directories (Kaggle Input)
KAGGLE_DATASET_ROOT="/kaggle/input/datasets/mngochocsupham/daisee"
# Thư mục chứa các file CSV nhãn
LABELS_DIR="$KAGGLE_DATASET_ROOT/DAiSEE_data/Labels"
# Writable directory for annotations on Kaggle
ANNOT_DIR="./daisee_annotations"

# Ensure annotation directory exists
mkdir -p "$ANNOT_DIR"

# --- Step 1: Generate Annotation Files from CSV ---
echo "=> Generating DAiSEE Annotations from CSV files..."

python3 - <<EOF
import os
import glob
import json
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed (though it is standard on Kaggle)
    def tqdm(iterable, **kwargs):
        return iterable

root_kaggle_dir = "$KAGGLE_DATASET_ROOT"
labels_dir = "$LABELS_DIR"
target_dir = "$ANNOT_DIR"

# 1. Quét toàn bộ dataset để tạo bản đồ ClipID -> Path thực tế
print("Step 1/3: Scanning disk for video files...")
clip_path_map = {}
# os.walk can be slow, using a counter to show some activity
file_count = 0
for root, dirs, files in os.walk(root_kaggle_dir):
    # Ưu tiên folder 'frames' nếu có (dành cho bản đã extract)
    if os.path.basename(root) == 'frames':
        clip_id = os.path.basename(os.path.dirname(root))
        clip_path_map[clip_id] = os.path.relpath(root, root_kaggle_dir)
        continue
    # Tìm các file video
    for file in files:
        if file.lower().endswith(('.avi', '.mp4', '.mov')):
            clip_id = os.path.splitext(file)[0]
            if clip_id not in clip_path_map:
                clip_path_map[clip_id] = os.path.relpath(os.path.join(root, file), root_kaggle_dir)
                file_count += 1
                if file_count % 1000 == 0:
                    print(f"  - Indexed {file_count} files...")

print(f"Total items indexed: {len(clip_path_map)}")

# 2. Parse các file CSV
# Sử dụng đường dẫn tuyệt đối trực tiếp từ user
csv_files = {
    "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels/TrainLabels.csv": "train.txt",
    "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels/ValidationLabels.csv": "val.txt",
    "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels/TestLabels.csv": "test.txt"
}

print("Step 2/3: Parsing CSV labels and mapping to paths...")
for csv_path, dst_name in csv_files.items():
    if not os.path.exists(csv_path):
        print(f"Warning: Label file {csv_path} not found.")
        continue
        
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    fixed_lines = []
    found_count = 0
    
    # Thêm Progress Bar ở đây
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dst_name}"):
        full_clip_name = str(row['ClipID']).strip()
        clip_id = os.path.splitext(full_clip_name)[0]
        
        # Lấy nhãn Engagement (0, 1, 2, 3)
        engagement = int(row['Engagement'])
        label_id = engagement + 1 # Chuyển sang 1-based (1, 2, 3, 4)
        
        # Tìm đường dẫn thực tế
        actual_path = clip_path_map.get(clip_id)
        
        if actual_path:
            fixed_lines.append(f"{actual_path} 300 {label_id}\n")
            found_count += 1
            
    with open(os.path.join(target_dir, dst_name), 'w') as f:
        f.writelines(fixed_lines)
    print(f"Finished {dst_name}: Matched {found_count}/{len(df)} samples.")

# 3. Tạo file dummy box
dummy_box_path = os.path.join(target_dir, "dummy_box.json")
with open(dummy_box_path, "w") as f:
    json.dump({}, f)
print("Step 3/3: Created dummy box file.")

print("Done! Annotations are ready.")
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
  --exper-name Train-DAiSEE-Kaggle-CSV \
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