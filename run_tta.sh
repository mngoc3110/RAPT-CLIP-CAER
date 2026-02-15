set -e
export OMP_NUM_THREADS=4

# Checkpoint path
checkpoint="model_best.pth"

echo "Running TTA with Confusion Bias on checkpoint: ${checkpoint}"

python eval_tta.py \
    --eval-checkpoint "${checkpoint}" \
    --gpu mps \
    --temporal-layers 1 \
    --batch-size 4 \
    --crop-body \
    --clip-path "ViT-B/16" \
    --confusion-bias 3 \
    --root-dir "./dataset" \
    --train-annotation "./dataset/RAER/annotation/train_80.txt" \
    --val-annotation "./dataset/RAER/annotation/val_20.txt" \
    --test-annotation "./dataset/RAER/annotation/test.txt" \
    --bounding-box-face "./dataset/RAER/bounding_box/face.json" \
    --bounding-box-body "./dataset/RAER/bounding_box/body.json"