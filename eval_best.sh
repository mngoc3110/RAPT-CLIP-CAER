set -e
export OMP_NUM_THREADS=4

# Config
# Update this path to your actual best checkpoint!
checkpoint_path="model_best.pth" 
dataset="RAER"
root_dir="./dataset"
# Cung cấp đầy đủ đường dẫn cho cả 3 file annotation để tránh lỗi FileNotFoundError
train_annotation="./dataset/RAER/annotation/train_80.txt"
val_annotation="./dataset/RAER/annotation/val_20.txt"
test_annotation="./dataset/RAER/annotation/test.txt"

bounding_box_face="./dataset/RAER/bounding_box/face.json"
bounding_box_body="./dataset/RAER/bounding_box/body.json"

echo "Running Evaluation on Test Set using checkpoint: ${checkpoint_path}"

python main.py \
    --mode eval \
    --eval-checkpoint "${checkpoint_path}" \
    --dataset "${dataset}" \
    --gpu mps \
    --workers 4 \
    --root-dir "${root_dir}" \
    --train-annotation "${train_annotation}" \
    --val-annotation "${val_annotation}" \
    --test-annotation "${test_annotation}" \
    --clip-path ViT-B/16 \
    --bounding-box-face "${bounding_box_face}" \
    --bounding-box-body "${bounding_box_body}" \
    --batch-size 4 \
    --use-amp \
    --text-type prompt_ensemble \
    --temporal-layers 1 \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --crop-body