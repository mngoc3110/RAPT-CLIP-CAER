set -e
export OMP_NUM_THREADS=4

# Set up the ULTIMATE High Performance Experiment for RAER
# Focus: HARDCORE LDAM to fix Confusion class + Temporal 3 Layers
# Upgrades: Deeper Model, Extreme Margin, MoCo, FP32
exper_name="RAER-Ultimate-LDAM-Hardcore-T3"
dataset="RAER"

# --- Training Parameters ---
use_ldl="False" 
ldl_temperature=1.0
ldl_warmup=2 

# MoCo Settings
use_moco="True"
moco_k=2048
moco_m=0.999
moco_t=0.07

# Optimization
batch_size=8
epochs=40 
lr=0.0001
lr_image_encoder=1e-06
lr_prompt_learner=0.001
lr_adapter=0.0001
weight_decay=0.1

# Paths
root_dir="./dataset/RAER"
train_annotation="./dataset/RAER/annotation/train_80.txt"
val_annotation="./dataset/RAER/annotation/val_20.txt"
test_annotation="./dataset/RAER/annotation/test.txt"
bounding_box_face="./dataset/RAER/bounding_box/face.json"
bounding_box_body="./dataset/RAER/bounding_box/body.json"

echo "Starting ULTIMATE Training: ${exper_name}"
echo "Upgrades: Temporal Layers=3, LDAM Margin=0.8, MoCo=Enabled, AMP=DISABLED"

# Note: Removed --use-amp to force FP32
python main.py \
    --mode train \
    --exper-name "${exper_name}" \
    --dataset "${dataset}" \
    --gpu 0 \
    --workers 4 \
    --seed 42 \
    --root-dir "${root_dir}" \
    --train-annotation "${train_annotation}" \
    --val-annotation "${val_annotation}" \
    --test-annotation "${test_annotation}" \
    --clip-path ViT-B/16 \
    --bounding-box-face "${bounding_box_face}" \
    --bounding-box-body "${bounding_box_body}" \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    --print-freq 50 \
    --optimizer AdamW \
    --lr ${lr} \
    --lr-image-encoder ${lr_image_encoder} \
    --lr-prompt-learner ${lr_prompt_learner} \
    --lr-adapter ${lr_adapter} \
    --weight-decay ${weight_decay} \
    --momentum 0.9 \
    --milestones 20 30 \
    --gamma 0.1 \
    --loss-type ldam \
    --ldam-max-m 0.8 \
    --ldam-s 30.0 \
    --lambda_mi 0.05 \
    --lambda_dc 0.05 \
    --use-weighted-sampler \
    --mixup-alpha 0.0 \
    --text-type prompt_ensemble \
    --temporal-layers 3 \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07 \
    --crop-body \
    --use-moco \
    --moco-k ${moco_k} \
    --moco-m ${moco_m} \
    --moco-t ${moco_t}