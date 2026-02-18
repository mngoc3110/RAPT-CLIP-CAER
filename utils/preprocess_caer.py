import os
import glob
import json
import argparse
import pandas as pd
import ast

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CAER-S dataset: Generate annotation files and bounding box JSON from CSVs.")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory of CAER-S dataset containing 'train', 'val', 'test' folders.")
    parser.add_argument('--output_dir', type=str, default='CAER_S_annotations', help="Directory to save generated .txt and .json files.")
    return parser.parse_args()

def parse_csv_files(root_dir, subsets, class_map):
    """
    Parses CSV files in the specified subsets (e.g., ['train', 'val']) directories.
    Returns a list of annotation entries and a dictionary of bounding boxes.
    """
    annotations = []
    bboxes = {} # Key: relative_path, Value: [x1, y1, x2, y2] 
    
    # Debug: Check if root_dir exists
    if not os.path.exists(root_dir):
        print(f"Error: Root directory '{root_dir}' does not exist.")
        return [], {}

    # Define standard class names for mapping
    classes_std = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    class_name_map = {c.lower(): c for c in classes_std}
    
    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)
        if not os.path.exists(subset_dir):
            continue
            
        # Find all CSV files in this subset directory (e.g., bbox_train_Angry.csv)
        # Search recursively to catch files in subfolders
        csv_files = glob.glob(os.path.join(subset_dir, "**", "*.csv"), recursive=True)

        print(f"  Found {len(csv_files)} CSV files in {subset}...")

        for csv_file in csv_files:
            try:
                # Read CSV, assuming no header or standard header
                # Based on user description: image_name, image_path, class, bbox
                # Let's try reading with header inference
                df = pd.read_csv(csv_file)
                
                # Verify columns exist
                required_cols = ['image_name', 'class', 'bbox']
                if not all(col in df.columns for col in required_cols):
                    print(f"    Warning: Missing columns in {csv_file}. Expected {required_cols}. Found {df.columns}. Skipping.")
                    continue
                
                for _, row in df.iterrows():
                    img_name = str(row['image_name']).strip()
                    class_name = str(row['class']).strip()
                    bbox_str = str(row['bbox']).strip()
                    
                    # Normalize class name
                    # CSV class might be 'Angry', directory might be 'Angry'
                    # Map to standard class name (Anger, Disgust, etc.)
                    # Note: CAER usually uses 'Anger' but some CSVs might use 'Angry'.
                    # Let's map 'Angry' -> 'Anger' if needed.
                    if class_name.lower() == 'angry':
                        class_name = 'Anger'
                    elif class_name.lower() in class_name_map:
                         class_name = class_name_map[class_name.lower()]
                    
                    if class_name not in class_map:
                        continue # Skip unknown class

                    label_idx = class_map[class_name]
                    
                    # Construct relative path
                    # Structure: subset/class_name/image_name
                    # Example: train/Angry/6017.png
                    # Note: Use the normalized class_name for folder structure?
                    # Or use the class name from CSV? Usually folder matches class.
                    # Let's assume folder structure follows Standard Class Names (Anger, Disgust...)
                    rel_path = os.path.join(subset, class_name, img_name)
                    
                    # Check if file exists using standard class name folder
                    full_path = os.path.join(root_dir, rel_path)
                    if not os.path.exists(full_path):
                        # Try the raw class name from CSV (e.g. 'Angry' folder?)
                        rel_path_alt = os.path.join(subset, row['class'].strip(), img_name)
                        full_path_alt = os.path.join(root_dir, rel_path_alt)
                        if os.path.exists(full_path_alt):
                            rel_path = rel_path_alt
                        else:
                            # print(f"    Warning: Image {rel_path} (and {rel_path_alt}) not found. Skipping.")
                            continue

                    # Parse Bounding Box
                    try:
                        # bbox is string "[x, y, w, h]" or "[x1, y1, x2, y2]"?
                        # User example: [227, 20, 548, 396] for 6017.png
                        # Usually this is [x1, y1, x2, y2]
                        bbox = ast.literal_eval(bbox_str)
                        if isinstance(bbox, list) and len(bbox) == 4:
                            bboxes[rel_path] = bbox
                    except:
                        print(f"    Warning: Could not parse bbox '{bbox_str}' for {rel_path}")

                    # Add to annotations: path num_frames label
                    annotations.append(f"{rel_path} 1 {label_idx}")
            
            except Exception as e:
                print(f"  Error processing {csv_file}: {e}")

    return annotations, bboxes

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define Classes (Standard CAER-S 7 classes)
    classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # Label mapping (1-based to match project convention)
    class_map = {cls: i+1 for i, cls in enumerate(classes)}
    
    print(f"Class Mapping: {class_map}")
    
    all_bboxes = {}

    # 1. Process Train & Val (Combine into train.txt)
    train_subsets = ['train', 'val', 'valid', 'training', 'validation']
    print("Processing Training sets...")
    train_anns, train_bboxes = parse_csv_files(args.root_dir, train_subsets, class_map)
    
    with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_anns))
    all_bboxes.update(train_bboxes)
    print(f"Saved {len(train_anns)} training entries.")

    # 2. Process Test (test.txt)
    test_subsets = ['test', 'testing']
    print("Processing Test sets...")
    test_anns, test_bboxes = parse_csv_files(args.root_dir, test_subsets, class_map)
    
    with open(os.path.join(args.output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_anns))
    all_bboxes.update(test_bboxes)
    print(f"Saved {len(test_anns)} testing entries.")

    # 3. Save Master Bounding Box JSON
    json_path = os.path.join(args.output_dir, 'caer_s_faces.json')
    with open(json_path, 'w') as f:
        json.dump(all_bboxes, f)
    print(f"Saved {len(all_bboxes)} bounding boxes to {json_path}")

    print("-" * 30)
    print("Preprocessing complete.")
    print(f"Please update your training script:")
    print(f"  --train-annotation {os.path.abspath(os.path.join(args.output_dir, 'train.txt'))}")
    print(f"  --val-annotation {os.path.abspath(os.path.join(args.output_dir, 'test.txt'))}")
    print(f"  --test-annotation {os.path.abspath(os.path.join(args.output_dir, 'test.txt'))}")
    print(f"  --bounding-box-face {os.path.abspath(json_path)}")

if __name__ == "__main__":
    main()