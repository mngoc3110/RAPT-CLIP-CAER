#!/bin/bash

# =================================================================
# DAiSEE ENGAGEMENT PIPELINE (KAGGLE SMART VERSION)
# =================================================================

# Writable directory for annotations on Kaggle
ANNOT_DIR="./daisee_annotations"
mkdir -p "$ANNOT_DIR"

echo "=> Generating DAiSEE Annotations from CSV labels..."

python3 - <<EOF
import os
import pandas as pd
import json

# Các đường dẫn CSV nhãn trên Kaggle của bạn
csv_mapping = {
    "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels/TrainLabels.csv": "train.txt",
    "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels/ValidationLabels.csv": "val.txt",
    "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data/Labels/TestLabels.csv": "test.txt"
}

for csv_path, dst_name in csv_mapping.items():
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found!")
        continue
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    lines = []
    for _, row in df.iterrows():
        clip_full = str(row['ClipID']).strip()
        # Lưu ClipID nguyên bản (có thể là 1100011002.avi)
        # Dataloader sẽ dùng ID này để tự tìm file trên đĩa
        
        # Nhãn Engagement (0-3) -> (1-4)
        label = int(row['Engagement']) + 1
        
        # Format: ID_hoặc_Path Số_frame Nhãn
        # Để số frame giả định là 300, dataloader sẽ tự đếm lại
        lines.append(f"{clip_full} 300 {label}\n")
        
    with open(os.path.join("$ANNOT_DIR", dst_name), 'w') as f:
        f.writelines(lines)
    print(f"Prepared {dst_name} with {len(lines)} samples.")

# Tạo dummy box
with open(os.path.join("$ANNOT_DIR", "dummy_box.json"), "w") as f:
    json.dump({}, f)
EOF

echo "=> Starting Training with DAiSEE Smart Dataloader..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --root-dir: Trỏ vào thư mục gốc của dataset trên Kaggle
# Dataloader sẽ tìm kiếm file bên trong thư mục này.
python main.py \
  --mode train \
  --exper-name Train-DAiSEE-Smart \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
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
  --root-dir "/kaggle/input/datasets/mngochocsupham/daisee/" \
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