import os
import argparse
from collections import Counter
import glob

def check_dataset(root_dir):
    print(f"Checking dataset at: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    # Strategy 1: Subfolders as classes
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    
    total_images = 0
    class_counts = {}
    
    print(f"Found {len(classes)} potential classes (subdirectories).")
    
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        count = 0
        for ext in extensions:
            count += len(glob.glob(os.path.join(cls_dir, ext)))
            count += len(glob.glob(os.path.join(cls_dir, ext.upper()))) # Case insensitive check
        
        class_counts[cls] = count
        total_images += count
        print(f"  Class '{cls}': {count} images")

    print("-" * 30)
    print(f"Total Images: {total_images}")
    print("-" * 30)
    
    if total_images == 0:
        print("No images found in subdirectories. Checking recursively...")
        # Recursive check just to be safe
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        print(f"Total recursive image count: {len(all_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset image counts.")
    parser.add_argument("--path", type=str, default=".", help="Path to the dataset root directory.")
    args = parser.parse_args()
    
    check_dataset(args.path)
