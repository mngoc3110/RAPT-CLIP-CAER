#!/bin/bash

# =================================================================
# DAiSEE OPTIMIZED PIPELINE (FULL DATA + WEIGHTED SAMPLER)
# =================================================================

# Define Annotation Directory
ANNOT_DIR="./daisee_annotations"
mkdir -p "$ANNOT_DIR"

echo "=> Generating Full DAiSEE Annotations (No Subsampling)..."

python3 - <<EOF
import os
import pandas as pd
import json

# Define paths to your CSV labels
csv_mapping = {
    "./dataset/DAiSEE/Labels/TrainLabels.csv": "train.txt",
    "./dataset/DAiSEE/Labels/ValidationLabels.csv": "val.txt",
    "./dataset/DAiSEE/Labels/TestLabels.csv": "test.txt"
}

for csv_path, dst_name in csv_mapping.items():
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found!")
        continue
    
    # Read CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # Remove spaces from column names
    
    lines = []
    for _, row in df.iterrows():
        clip_full = str(row['ClipID']).strip()
        # Label mapping: 0->1, 1->2, 2->3, 3->4 (to match 1-based indexing if needed)
        # Assuming original labels are 0,1,2,3. Adjust if they are already 1-4.
        # Project convention seems to be 0-based in code, but some dataloaders subtract 1.
        # Let's assume standard 0-3 and add 1 for safety if dataloader subtracts 1.
        label = int(row['Engagement']) + 1 
        
        # Format: Path NumFrames Label
        lines.append(f"{clip_full} 300 {label}\n")
        
    with open(os.path.join("$ANNOT_DIR", dst_name), 'w') as f:
        f.writelines(lines)
    print(f"Generated {dst_name} with {len(lines)} samples.")

# Create dummy box file (Dataloader will use full frame or center crop if box is empty/dummy)
with open(os.path.join("$ANNOT_DIR", "dummy_box.json"), "w") as f:
    json.dump({}, f)
EOF

echo "=> Starting Training (Video Mode)..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train-DAiSEE-Optimized \
  --dataset DAiSEE \
  --gpu mps \
  --epochs 20 \
  --batch-size 8 \
  --workers 2 \
  --optimizer AdamW \
  --lr 1e-4 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-3 \
  --lr-adapter 1e-4 \
  --weight-decay 0.1 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 2 \
  --num-segments 4 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "./dataset/DAiSEE/DataSet" \
  --train-annotation "$ANNOT_DIR/train.txt" \
  --val-annotation "$ANNOT_DIR/val.txt" \
  --test-annotation "$ANNOT_DIR/test.txt" \
  --clip-path ViT-B/16 \
  --bounding-box-face "$ANNOT_DIR/dummy_box.json" \
  --bounding-box-body "$ANNOT_DIR/dummy_box.json" \
  --text-type prompt_ensemble \
  --lambda_dc 0.05 \
  --lambda_mi 0.05 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --mixup-alpha 0.2 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --use-weighted-sampler \
  --temperature 0.7 \
  --use-amp \
  --grad-clip 1.0 \
  --use-ldl \
  --ldl-temperature 1.0
