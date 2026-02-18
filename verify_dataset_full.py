import os
import json
from collections import Counter

# Cấu hình đường dẫn
ROOT_DIR = "./CAER-S"
TRAIN_FILE = "caer_s_annotations/train.txt"
TEST_FILE = "caer_s_annotations/test.txt"
BBOX_FILE = "caer_s_annotations/caer_s_faces.json"

# Mapping nhãn theo chuẩn preprocess trước đó
LABEL_MAP = {
    "1": "Anger",
    "2": "Disgust",
    "3": "Fear",
    "4": "Happy",
    "5": "Neutral",
    "6": "Sad",
    "7": "Surprise"
}

def verify_subset(name, txt_path, bbox_data):
    print(f"\n{'='*20} VERIFYING: {name} {'='*20}")
    
    if not os.path.exists(txt_path):
        print(f"ERROR: File {txt_path} not found!")
        return

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    total_entries = len(lines)
    missing_files = 0
    missing_bboxes = 0
    label_counts = Counter()
    
    # Duyệt từng dòng
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        rel_path = parts[0]
        # frame_count = parts[1] (ignored)
        label = parts[2]
        
        # 1. Thống kê Label
        label_name = LABEL_MAP.get(label, f"Unknown({label})")
        label_counts[label_name] += 1
        
        # 2. Kiểm tra File ảnh tồn tại vật lý
        full_path = os.path.join(ROOT_DIR, rel_path)
        if not os.path.exists(full_path):
            missing_files += 1
            # print(f"Missing File: {full_path}") # Uncomment to debug
            
        # 3. Kiểm tra Bounding Box
        if rel_path not in bbox_data:
            missing_bboxes += 1

    # IN KẾT QUẢ
    print(f"Total Entries (Lines in .txt): {total_entries}")
    
    print("-" * 10 + " LABEL DISTRIBUTION " + "-" * 10)
    # Sắp xếp theo Label ID để dễ nhìn
    sorted_counts = sorted(label_counts.items(), key=lambda x: [k for k,v in LABEL_MAP.items() if v == x[0]][0])
    for label_name, count in sorted_counts:
        print(f"  {label_name:<10}: {count:>5}")
        
    print("-" * 10 + " INTEGRITY CHECK " + "-" * 10)
    print(f"  Missing Physical Files : {missing_files} (Files not found on disk)")
    if missing_files > 0:
        print("  => WARNING: Dataloader will crash if not handled!")
    else:
        print("  => OK: All files exist.")
        
    print(f"  Missing Bounding Boxes : {missing_bboxes} (Will fallback to Center Crop)")
    percent_bbox = ((total_entries - missing_bboxes) / total_entries) * 100 if total_entries > 0 else 0
    print(f"  => OK: {percent_bbox:.2f}% images have face coordinates.")

def main():
    print("Loading Bounding Box JSON...")
    if os.path.exists(BBOX_FILE):
        with open(BBOX_FILE, 'r') as f:
            bbox_data = json.load(f)
        print(f"Loaded {len(bbox_data)} bounding boxes.")
    else:
        print("ERROR: Bounding box file not found!")
        bbox_data = {}

    verify_subset("TRAIN SET (Includes Valid)", TRAIN_FILE, bbox_data)
    verify_subset("TEST SET", TEST_FILE, bbox_data)

if __name__ == "__main__":
    main()
