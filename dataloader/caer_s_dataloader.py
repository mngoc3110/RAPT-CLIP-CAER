import os
import glob
import cv2
import torch
import numpy as np
import random
from PIL import Image
from torch.utils import data
from dataloader.video_transform import *

class CAERSDataset(data.Dataset):
    def __init__(self, root_dir, mode, num_segments, duration, image_size, transform=None, val_split=0.15, seed=42, subsample_ratio=1.0):
        self.root_dir = root_dir
        self.mode = mode
        self.num_segments = num_segments
        self.duration = duration
        self.image_size = image_size
        self.transform = transform
        self.val_split = val_split
        self.seed = seed
        self.subsample_ratio = subsample_ratio

        # Classes for CAER-S
        self.class_map = {
            'Anger': 0, 'Angry': 0,
            'Disgust': 1,
            'Fear': 2,
            'Happy': 3,
            'Neutral': 4,
            'Sad': 5,
            'Surprise': 6
        }
        
        # Load samples
        self.samples = self._make_dataset(root_dir)
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 files in subfolders of: {root_dir}")

        # Face detector path
        self.cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

    def _make_dataset(self, directory):
        directory = os.path.expanduser(directory)
        
        # Logic xử lý mode: map to specific folder
        if self.mode == 'train':
            source_folder = 'train'
        elif self.mode == 'val':
            # Use dedicated validation folder if exists, otherwise try 'val'
            source_folder = 'validation'
            if not os.path.exists(os.path.join(directory, source_folder)) and os.path.exists(os.path.join(directory, 'val')):
                 source_folder = 'val'
        else: # mode == 'test'
            source_folder = 'test'
            
        target_dir = os.path.join(directory, source_folder)
        print(f"Scanning {target_dir} for CAER-S data (Source for {self.mode})...")
        
        if not os.path.exists(target_dir):
            raise RuntimeError(f"Directory not found: {target_dir}")

        # Collect samples per class to maintain balance when subsampling
        samples_per_class = {}
        
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            class_name = os.path.basename(root)
            if class_name in self.class_map:
                class_index = self.class_map[class_name]
                if class_index not in samples_per_class:
                    samples_per_class[class_index] = []
                    
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(root, fname)
                        samples_per_class[class_index].append((path, class_index))
        
        # Subsampling logic
        all_samples = []
        
        # Nếu subsample_ratio > 1, ta hiểu đó là số lượng mẫu cụ thể (samples_per_class)
        # Nếu < 1, ta hiểu đó là tỷ lệ phần trăm (như cũ)
        is_fixed_count = self.subsample_ratio > 1
        
        if self.subsample_ratio < 1.0 or is_fixed_count:
            if is_fixed_count:
                print(f"Limiting dataset to {int(self.subsample_ratio)} samples per class...")
            else:
                print(f"Subsampling dataset with ratio {self.subsample_ratio}...")
                
            random.seed(self.seed) # Ensure reproducible subsampling
            for class_idx, samples in samples_per_class.items():
                # Shuffle before picking to get random samples
                random.shuffle(samples)
                
                if is_fixed_count:
                    n_samples = min(len(samples), int(self.subsample_ratio))
                else:
                    n_samples = max(1, int(len(samples) * self.subsample_ratio))
                    
                selected = samples[:n_samples]
                all_samples.extend(selected)
                print(f"  Class {class_idx}: Selected {len(selected)}/{len(samples)} samples")
        else:
            for samples in samples_per_class.values():
                all_samples.extend(samples)
        
        # Shuffle final list
        random.seed(self.seed)
        random.shuffle(all_samples)
        
        print(f"Found {len(all_samples)} samples for {self.mode} set.")
        return all_samples

    def _face_detect_cv2(self, img_pil):
        # Convert PIL to CV2 (BGR)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Lazy initialization of CascadeClassifier
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Pick the largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            # Add some margin
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_pil.width - x, w + 2*margin)
            h = min(img_pil.height - y, h + 2*margin)
            
            face_img = img_pil.crop((x, y, x+w, y+h))
            return face_img
        else:
            # Fallback: Return original image if no face detected
            return img_pil

    def __getitem__(self, index):
        path, target = self.samples[index]
        
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return dummy
            dummy = torch.zeros((self.num_segments, 3, self.image_size, self.image_size))
            return dummy, dummy, target

        # Detect face
        face_img = self._face_detect_cv2(img)
        
        # Prepare lists for transform (simulate video by repeating)
        images_list = [img.copy() for _ in range(self.num_segments)]
        faces_list = [face_img.copy() for _ in range(self.num_segments)]
        
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

def caer_s_data_loader(root_dir, mode, num_segments, duration, image_size):
    # Cấu hình số lượng mẫu cứng mỗi nhãn
    if mode == 'train':
        SAMPLES_PER_CLASS = 500
    elif mode == 'val':
        SAMPLES_PER_CLASS = 200
    else: # test
        SAMPLES_PER_CLASS = None # Lấy toàn bộ tập Test để có kết quả chính xác nhất

    # Chúng ta truyền tham số này vào dataset thông qua biến subsample_ratio 
    # (nhưng sửa logic bên trong class Dataset để hiểu đây là số lượng thay vì tỷ lệ)
    
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

    dataset = CAERSDataset(
        root_dir=root_dir,
        mode=mode,
        num_segments=num_segments,
        duration=duration,
        image_size=image_size,
        transform=transform,
        subsample_ratio=SAMPLES_PER_CLASS # Tạm dùng biến này để truyền số lượng
    )
    return dataset
