# Kaggle Training Script for CAER-S
# Copy this entire block into a single code cell in your Kaggle Notebook.

# 1. Define Paths (Python variables)
dataset_root = "/kaggle/input/datasets/lcngtr/caer-s"
annotation_root = "/kaggle/working/annotations"
bbox_root = "/kaggle/working/bboxes"

# 2. Run Training
# We use {variable} syntax which Jupyter replaces with the Python variable value.
!python main.py \
  --mode train \
  --exper-name Train-CAER-SOTA-Colab \
  --dataset CAER-S \
  --gpu 0 \
  --epochs 3 \
  --batch-size 32 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 5e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.05 \
  --milestones 10 15 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir {dataset_root} \
  --train-annotation {annotation_root}/train.txt \
  --val-annotation {annotation_root}/test.txt \
  --test-annotation {annotation_root}/test.txt \
  --bounding-box-face {bbox_root}/face.json \
  --bounding-box-body {bbox_root}/body.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.35 \
  --lambda_dc 0.1 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --dc-warmup 5 \
  --use-amp \
  --use-weighted-sampler \
  --crop-body \
  --grad-clip 1.0 \
  --mixup-alpha 0.0
