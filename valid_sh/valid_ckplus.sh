#!/bin/bash

# =================================================================
# VALIDATION SCRIPT FOR CK+ (NO MOCO)
# =================================================================

# Define Data Directories
ROOT_DATA_DIR="./dataset/CKPlus_Dataset"
ANNOT_DIR="$ROOT_DATA_DIR/annotations"

# Default checkpoint path (Update this after training!)
DEFAULT_CHECKPOINT="outputs/Train-CKPlus-Flat-[02-10]-[08:44]/model_best.pth"

CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint file '$CHECKPOINT' not found."
    echo "Usage: bash valid_ckplus.sh path/to/model_best.pth"
fi

echo "========================================================"
echo "=> Starting Evaluation on CK+..."
echo "========================================================"

python main.py \
    --mode eval \
    --exper-name Eval-CKPlus \
    --dataset CK+ \
    --gpu mps \
    --eval-checkpoint "$CHECKPOINT" \
    --root-dir ./ \
    --train-annotation "$ANNOT_DIR/train.txt" \
    --val-annotation "$ANNOT_DIR/val.txt" \
    --test-annotation "$ANNOT_DIR/test.txt" \
    --clip-path ViT-B/16 \
    --bounding-box-face "$ANNOT_DIR/dummy_box.json" \
    --bounding-box-body "$ANNOT_DIR/dummy_box.json" \
    --text-type prompt_ensemble \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 1 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --temperature 0.07 \
    --crop-body
# Note: --use-moco removed to match training configuration