#!/bin/bash

# =================================================================
# DAiSEE ENGAGEMENT PIPELINE (KAGGLE BALANCED LITE - STABLE)
# =================================================================

# Writable directory for annotations on Kaggle
ANNOT_DIR="./daisee_annotations"
mkdir -p "$ANNOT_DIR"

# Số lượng mẫu tối đa cho MỖI lớp trong tập Train.
MAX_SAMPLES_PER_CLASS=500

echo "=> Generating DAiSEE Annotations (Balanced Subsample: $MAX_SAMPLES_PER_CLASS per class)..."

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

# Pass bash variable to python
max_samples = $MAX_SAMPLES_PER_CLASS

for csv_path, dst_name in csv_mapping.items():
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found!")
        continue
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Balanced Subsampling only for Training set
    if "Train" in csv_path:
        print(f"Balancing {dst_name} with max {max_samples} samples per class...")
        
        dfs = []
        for label in sorted(df['Engagement'].unique()):
            df_class = df[df['Engagement'] == label]
            if len(df_class) > max_samples:
                df_class = df_class.sample(n=max_samples, random_state=42)
            dfs.append(df_class)
            print(f"  - Class {label}: {len(df_class)} samples")
            
        df = pd.concat(dfs).sample(frac=1, random_state=42) # Shuffle
        print(f"  -> Total Balanced Train Samples: {len(df)}")
    
    lines = []
    for _, row in df.iterrows():
        clip_full = str(row['ClipID']).strip()
        label = int(row['Engagement']) + 1
        lines.append(f"{clip_full} 300 {label}\n")
        
    with open(os.path.join("$ANNOT_DIR", dst_name), 'w') as f:
        f.writelines(lines)
    print(f"Prepared {dst_name} with {len(lines)} samples.")

# Tạo dummy box
with open(os.path.join("$ANNOT_DIR", "dummy_box.json"), "w") as f:
    json.dump({}, f)
EOF

echo "=> Starting Training with Stable Hyperparameters..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cấu hình ổn định: LR thấp hơn, tắt LDL, tăng WD.
python main.py \
  --mode train \
  --exper-name Train-DAiSEE-Stable \
  --dataset DAiSEE \
  --gpu mps \
  --epochs 20 \
  --batch-size 4 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 2e-5 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 8 \
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
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --lambda_dc 0.1 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --temperature 0.07 \
  --use-moco \
  --moco-k 4096 \
  --moco-m 0.99 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.2