import json
import os
import cv2
import mediapipe as mp
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate/Fix bounding boxes for CAER-S using MediaPipe.")
    parser.add_argument('--root-dir', type=str, default='./CAER-S', help="Path to CAER-S dataset root.")
    parser.add_argument('--ann-dir', type=str, default='./CAER-S/annotations', help="Path to annotations folder.")
    return parser.parse_args()

def main():
    args = parse_args()
    bbox_path = os.path.join(args.ann_dir, 'caer_s_faces.json')
    
    # Load existing bboxes if any
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r') as f:
            bboxes = json.load(f)
        print(f"Loaded {len(bboxes)} existing bounding boxes from {bbox_path}.")
    else:
        bboxes = {}
        print(f"Starting with empty bounding box dictionary. Will save to {bbox_path}.")

    # Get all paths from train.txt and test.txt (and validation.txt if present)
    all_paths = []
    for split in ['train.txt', 'test.txt', 'validation.txt']:
        p = os.path.join(args.ann_dir, split)
        if os.path.exists(p):
            with open(p, 'r') as f:
                for line in f:
                    # Format: path label
                    path = line.strip().split()[0]
                    all_paths.append(path)
    
    missing_paths = [p for p in all_paths if p not in bboxes]
    print(f"Detected {len(missing_paths)} missing face coordinates.")
    
    if not missing_paths:
        print("No missing coordinates found. Exiting.")
        return

    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)

    fixed_count = 0
    fail_count = 0
    
    print("Processing images with MediaPipe...")
    for rel_path in tqdm(missing_paths):
        full_path = os.path.join(args.root_dir, rel_path)
        if not os.path.exists(full_path):
            continue
            
        img = cv2.imread(full_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            # Take the detection with highest confidence (usually the main person)
            detection = max(results.detections, key=lambda d: d.score[0])
            bb = detection.location_data.relative_bounding_box
            
            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            w_box = int(bb.width * w)
            h_box = int(bb.height * h)
            
            # Convert to [x1, y1, x2, y2]
            coords = [
                max(0, x1),
                max(0, y1),
                min(w, x1 + w_box),
                min(h, y1 + h_box)
            ]
            bboxes[rel_path] = coords
            fixed_count += 1
        else:
            fail_count += 1

    # Save updated JSON
    with open(bbox_path, 'w') as f:
        json.dump(bboxes, f)
        
    print(f"\nFinished!")
    print(f"  - Successfully fixed/added: {fixed_count}")
    print(f"  - Still missing (detection failed): {fail_count}")
    print(f"  - Total bounding boxes in JSON: {len(bboxes)}")

if __name__ == "__main__":
    main()
