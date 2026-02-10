import os
import glob
import shutil
import random
import json

# Configuration
SOURCE_DIR = 'ckplus/CK+48'
TARGET_DIR = 'ckplus_ready'
ANNOTATION_DIR = 'ckplus_annotation'
TRAIN_RATIO = 0.8

# CK+ Classes (sorted alphabetically based on folder names)
# Note: The folder names in CK+48/ are usually: anger, contempt, disgust, fear, happy, sadness, surprise
classes = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])
print(f"Found classes: {classes}")

class_to_idx = {cls_name: i+1 for i, cls_name in enumerate(classes)} # 1-based index for dataloader

# Prepare directories
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
os.makedirs(TARGET_DIR)

if not os.path.exists(ANNOTATION_DIR):
    os.makedirs(ANNOTATION_DIR)

train_lines = []
test_lines = []

for cls_name in classes:
    cls_dir = os.path.join(SOURCE_DIR, cls_name)
    images = glob.glob(os.path.join(cls_dir, '*.png'))
    
    # Group by sequence (Subject_Sequence)
    sequences = {}
    for img_path in images:
        filename = os.path.basename(img_path)
        # S010_004_00000017.png -> S010_004
        parts = filename.split('_')
        seq_id = f"{parts[0]}_{parts[1]}"
        
        if seq_id not in sequences:
            sequences[seq_id] = []
        sequences[seq_id].append(img_path)
    
    # Process sequences
    seq_ids = list(sequences.keys())
    random.shuffle(seq_ids)
    
    split_idx = int(len(seq_ids) * TRAIN_RATIO)
    train_seqs = seq_ids[:split_idx]
    test_seqs = seq_ids[split_idx:]
    
    for seq_id in seq_ids:
        # Create output folder for this video sequence
        # Structure: ckplus_ready/S010_004
        video_dir_name = seq_id
        video_output_path = os.path.join(TARGET_DIR, video_dir_name)
        os.makedirs(video_output_path, exist_ok=True)
        
        # Copy frames
        frames = sorted(sequences[seq_id])
        for frame_path in frames:
            shutil.copy(frame_path, video_output_path)
            
        # Create annotation line
        # Format: path num_frames label_id
        # Note: path should be relative to root_dir in dataloader, which we will set to './'
        # So path is ckplus_ready/S010_004
        
        num_frames = len(frames)
        label = class_to_idx[cls_name]
        line = f"{video_output_path} {num_frames} {label}\n"
        
        if seq_id in train_seqs:
            train_lines.append(line)
        else:
            test_lines.append(line)

# Write annotation files
train_file = os.path.join(ANNOTATION_DIR, 'train.txt')
test_file = os.path.join(ANNOTATION_DIR, 'test.txt')

with open(train_file, 'w') as f:
    f.writelines(train_lines)

with open(test_file, 'w') as f:
    f.writelines(test_lines)

# Create dummy empty JSON for bounding boxes
# The dataloader handles keys not found by returning the full image, which is what we want for CK+
dummy_json_path = os.path.join(ANNOTATION_DIR, 'dummy_box.json')
with open(dummy_json_path, 'w') as f:
    json.dump({}, f)

print(f"Processed {len(train_lines)} training sequences and {len(test_lines)} testing sequences.")
print(f"Data ready in {TARGET_DIR}")
print(f"Annotations in {ANNOTATION_DIR}")
print(f"Class mapping: {class_to_idx}")
