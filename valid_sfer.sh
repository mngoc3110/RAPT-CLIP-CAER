#!/bin/bash

# =================================================================
# VALIDATION SCRIPT FOR SFER
# =================================================================

# Define Data Directories
ROOT_DATA_DIR="./Student_FER_Dataset"
ANNOT_DIR="$ROOT_DATA_DIR/annotations"

# Default checkpoint path (Update this after training!)
# This is a placeholder; user should provide the path or update it.
DEFAULT_CHECKPOINT="outputs/Train-SFER-[DATE]-[TIME]/model_best.pth"

CHECKPOINT="${1:-$DEFAULT_CHECKPOINT}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint file '$CHECKPOINT' not found."
    echo "Usage: bash valid_sfer.sh path/to/model_best.pth"
fi

echo "========================================================"
echo "=> Starting Evaluation on SFER..."
echo "========================================================"

python main.py \
    --mode eval \
    --exper-name Eval-SFER \
    --dataset SFER \
    --gpu 0 \
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
