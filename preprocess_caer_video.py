import os
import cv2
import json
import mediapipe as mp
from tqdm import tqdm

ROOT = "./CAER_Video"
OUTPUT_DIR = "caer_video_annotations"
CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
class_map = {cls: i for i, cls in enumerate(CLASSES)}

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def process_set(subset):
    anns = []
    bboxes = {}
    subset_path = os.path.join(ROOT, subset)
    
    if not os.path.exists(subset_path):
        print(f"Warning: Subset path {subset_path} not found.")
        return [], {}

    for cls in CLASSES:
        cls_path = os.path.join(subset_path, cls)
        if not os.path.exists(cls_path): 
            continue
        
        videos = sorted([f for f in os.listdir(cls_path) if f.endswith('.avi')])
        print(f"Processing {subset}/{cls} ({len(videos)} videos)...")
        
        for vid_name in tqdm(videos):
            vid_path = os.path.join(cls_path, vid_name)
            # Use forward slashes for cross-platform compatibility in annotation files
            rel_path = f"{subset}/{cls}/{vid_name}"
            
            cap = cv2.VideoCapture(vid_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if num_frames > 0:
                # Detect face in the middle frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, num_frames // 2)
                ret, frame = cap.read()
                if ret:
                    h_img, w_img, _ = frame.shape
                    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.detections:
                        # Get first detection (usually the main person)
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        
                        x1 = int(bbox.xmin * w_img)
                        y1 = int(bbox.ymin * h_img)
                        w_box = int(bbox.width * w_img)
                        h_box = int(bbox.height * h_img)
                        
                        # Clip coordinates to image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w_img, x1 + w_box)
                        y2 = min(h_img, y1 + h_box)
                        
                        # Store as [x1, y1, x2, y2]
                        bboxes[rel_path] = [x1, y1, x2, y2]
                
                anns.append(f"{rel_path} {num_frames} {class_map[cls]}")
            cap.release()
            
    return anns, bboxes

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_bboxes = {}
    
    for subset in ['train', 'validation', 'test']:
        anns, bboxes = process_set(subset)
        if anns:
            output_file = os.path.join(OUTPUT_DIR, f"{subset}.txt")
            with open(output_file, 'w') as f:
                f.write("\n".join(anns))
            print(f"Saved {len(anns)} annotations to {output_file}")
        all_bboxes.update(bboxes)

    bbox_output = os.path.join(OUTPUT_DIR, "caer_video_faces.json")
    with open(bbox_output, 'w') as f:
        json.dump(all_bboxes, f)
    print(f"Saved {len(all_bboxes)} face bounding boxes to {bbox_output}")

if __name__ == "__main__":
    main()
