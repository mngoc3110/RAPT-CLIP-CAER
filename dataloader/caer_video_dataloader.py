import os
import glob
import cv2
import torch
import torchvision
import numpy as np
import random
from PIL import Image
from torch.utils import data
from dataloader.video_transform import *

class CAERVideoDataset(data.Dataset):
    def __init__(self, root_dir, mode, num_segments, duration, image_size, transform=None, val_split=0.15, seed=42, samples_per_class=None):
        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.duration = duration
        self.image_size = image_size
        self.transform = transform
        self.val_split = val_split
        self.seed = seed
        self.samples_per_class = samples_per_class

        # Classes for CAER (Same as CAER-S generally, but let's be explicit)
        self.class_map = {
            'Anger': 0, 'Angry': 0,
            'Disgust': 1,
            'Fear': 2,
            'Happy': 3,
            'Neutral': 4,
            'Sad': 5,
            'Surprise': 6
        }
        
        # Load video paths
        self.samples = self._make_dataset(root_dir)
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 video files in subfolders of: {root_dir}")

        # Face detector path (optional, but good for consistency)
        self.cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

    def _make_dataset(self, directory):
        directory = os.path.expanduser(directory)
        
        # Handle mode: Map mode to folder name
        if self.mode == 'train':
            source_folder = 'train'
        elif self.mode == 'val':
            # Use the dedicated validation folder if available
            # Note: CAER dataset has 'validation' folder
            source_folder = 'validation'
            # Fallback if 'validation' doesn't exist but 'val' does (just in case)
            if not os.path.exists(os.path.join(directory, source_folder)) and os.path.exists(os.path.join(directory, 'val')):
                source_folder = 'val'
        else: # mode == 'test'
            source_folder = 'test'
            
        target_dir = os.path.join(directory, source_folder)
        print(f"Scanning {target_dir} for CAER video data (Source for {self.mode})...")
        
        if not os.path.exists(target_dir):
             raise RuntimeError(f"Directory not found: {target_dir}")

        # Collect samples per class
        samples_per_class_dict = {}
        
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            class_name = os.path.basename(root)
            if class_name in self.class_map:
                class_index = self.class_map[class_name]
                if class_index not in samples_per_class_dict:
                    samples_per_class_dict[class_index] = []
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.avi', '.mp4', '.mov')):
                        path = os.path.join(root, fname)
                        samples_per_class_dict[class_index].append((path, class_index))
        
        # Limit samples per class if specified
        all_samples = []
        if self.samples_per_class is not None:
            print(f"Limiting dataset to {self.samples_per_class} videos per class...")
            random.seed(self.seed)
            for class_idx, samples in samples_per_class_dict.items():
                # Only shuffle and limit if we need to subsample
                # If we want deterministic full set (like test), we might not shuffle, but here we enforce limit
                random.shuffle(samples)
                selected = samples[:self.samples_per_class]
                all_samples.extend(selected)
                print(f"  Class {class_idx}: Selected {len(selected)}/{len(samples)} videos")
        else:
            for samples in samples_per_class_dict.values():
                all_samples.extend(samples)
        
        # Shuffle final list for training, keep sorted/stable for val/test if desired?
        # Actually DataLoader handles shuffle for train. 
        # But for 'val' and 'test', usually we don't need to shuffle inside dataset unless we subsampled.
        # Let's shuffle all to mix classes in the list (good for batch norm if not shuffling loader)
        random.seed(self.seed)
        random.shuffle(all_samples)
        
        print(f"Found {len(all_samples)} videos for {self.mode} set.")
        return all_samples

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def _face_detect_cv2(self, img_pil):
        # Convert PIL to CV2 (BGR)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_pil.width - x, w + 2*margin)
            h = min(img_pil.height - y, h + 2*margin)
            face_img = img_pil.crop((x, y, x+w, y+h))
            return face_img
        else:
            return img_pil

    def _sample_indices(self, num_frames):
        if num_frames == 0:
            return []
            
        average_duration = (num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(np.random.randint(num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(num_frames))), (0, self.num_segments - num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        path, target = self.samples[index]
        
        frames = self._load_video(path)
        
        if len(frames) == 0:
            print(f"Warning: Video {path} is empty.")
            dummy = torch.zeros((self.num_segments, 3, self.image_size, self.image_size))
            return dummy, dummy, target

        indices = self._sample_indices(len(frames))
        
        images_list = []
        faces_list = []
        
        for i in indices:
            # Safe indexing
            idx = min(i, len(frames)-1)
            img = frames[idx]
            
            # Detect face
            face_img = self._face_detect_cv2(img)
            
            images_list.append(img)
            faces_list.append(face_img)

        # Apply transforms
        if self.transform is not None:
            images_trans = self.transform(images_list)
            faces_trans = self.transform(faces_list)
        else:
            t = ToTorchFormatTensor()
            images_trans = t(np.concatenate([np.array(x) for x in images_list], axis=2))
            faces_trans = t(np.concatenate([np.array(x) for x in faces_list], axis=2))

        images_trans = torch.reshape(images_trans, (-1, 3, self.image_size, self.image_size))
        faces_trans = torch.reshape(faces_trans, (-1, 3, self.image_size, self.image_size))
        
        return faces_trans, images_trans, target

    def __len__(self):
        return len(self.samples)

def caer_video_data_loader(root_dir, mode, num_segments, duration, image_size):
    # Cấu hình số lượng mẫu cứng mỗi nhãn
    if mode == 'train':
        SAMPLES_PER_CLASS = 500
    elif mode == 'val':
        SAMPLES_PER_CLASS = 200
    else: # test
        SAMPLES_PER_CLASS = None # Lấy toàn bộ tập Test để có kết quả chính xác nhất

    if mode == 'train':
        transform = torchvision.transforms.Compose([
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()
        ])
    else: # val or test
        transform = torchvision.transforms.Compose([
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor()
        ])

    dataset = CAERVideoDataset(
        root_dir=root_dir,
        mode=mode,
        num_segments=num_segments,
        duration=duration,
        image_size=image_size,
        transform=transform,
        samples_per_class=SAMPLES_PER_CLASS
    )
    return dataset
