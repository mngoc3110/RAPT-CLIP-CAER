import os
import glob
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloader.video_transform import GroupResize, Stack, ToTorchFormatTensor

class CAERSDataset(data.Dataset):
    def __init__(self, root_dir, list_file, mode='train', image_size=224, num_classes=7, bounding_box_json=None):
        """
        Args:
            root_dir (str): Base path to the dataset images.
            list_file (str): Path to the annotation file (e.g., 'train.txt').
            mode (str): 'train', 'val', or 'test'.
            image_size (int): Size to resize images to.
            bounding_box_json (str): Path to JSON file containing face bounding boxes.
        """
        self.root_dir = root_dir
        self.list_file = list_file
        self.mode = mode
        self.image_size = image_size
        self.num_classes = num_classes
        self.bounding_box_json = bounding_box_json
        
        # Load Bounding Boxes
        self.bboxes = {}
        if self.bounding_box_json and os.path.exists(self.bounding_box_json):
            try:
                with open(self.bounding_box_json, 'r') as f:
                    self.bboxes = json.load(f)
                print(f"Loaded {len(self.bboxes)} bounding boxes from {self.bounding_box_json}")
            except Exception as e:
                print(f"Error loading bounding box JSON: {e}")
        
        # CAER-S Classes (sorted alphabetically to match models/Text.py)
        # ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        self.samples = self._make_dataset()
        
        # Transforms
        # Standard ImageNet/CLIP preprocessing strategy
        if mode == 'train':
            # Stronger augmentation for training to prevent overfitting on static images
            self.transform_body = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), # Capture context but vary it
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
            self.transform_face = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)), # Tighter crop for face simulation
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            # Deterministic transform for validation/test
            self.transform_body = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)), # Resize to 256
                transforms.CenterCrop(image_size),              # Crop center 224
                transforms.ToTensor(),
            ])
            self.transform_face = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def _make_dataset(self):
        samples = []
        if not os.path.exists(self.list_file):
            print(f"Error: Annotation file {self.list_file} not found.")
            return samples
            
        with open(self.list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Format: path num_frames label
                    # path is relative to root_dir
                    rel_path = parts[0]
                    # num_frames = parts[1] (ignored for static image)
                    label = int(parts[2])
                    
                    full_path = os.path.join(self.root_dir, rel_path)
                    samples.append((full_path, label, rel_path)) # Store rel_path for bbox lookup
                elif len(parts) == 2:
                    # Handle case where num_frames is omitted: path label
                    rel_path = parts[0]
                    label = int(parts[1])
                    full_path = os.path.join(self.root_dir, rel_path)
                    samples.append((full_path, label, rel_path))
                else:
                     print(f"Warning: Skipping malformed line in {self.list_file}: {line.strip()}")
        
        print(f"Loaded {len(samples)} samples from {self.list_file}")
        return samples

    def _get_face_crop(self, img, rel_path):
        """
        Get face crop using bounding box if available, otherwise fallback to center crop heuristic.
        """
        # 1. Try JSON Bounding Box
        if rel_path in self.bboxes:
            bbox = self.bboxes[rel_path]
            # bbox format: [x1, y1, x2, y2]
            try:
                x1, y1, x2, y2 = bbox
                # Ensure coordinates are within image bounds
                w, h = img.size
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    # Add margin (e.g., 10%) to include some context around face
                    margin_x = (x2 - x1) * 0.1
                    margin_y = (y2 - y1) * 0.1
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)
                    
                    return img.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"Warning: Failed to crop bbox {bbox} for {rel_path}: {e}")

        # 2. Fallback: Heuristic Center Crop
        w, h = img.size
        # Crop central 50% of the image area (heuristic for "close-up")
        new_w, new_h = w * 0.5, h * 0.5
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        
        return img.crop((left, top, right, bottom))

    def __getitem__(self, index):
        path, label, rel_path = self.samples[index]
        
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = Image.new('RGB', (self.image_size, self.image_size))

        # 1. Body/Context Image (Full Image)
        img_body = img
        
        # 2. Face Image (JSON Bbox or Heuristic Center Crop)
        # This differentiates the input to the Face Adapter from the Body input
        img_face = self._get_face_crop(img, rel_path)
        
        # Apply transforms
        # Result is (C, H, W)
        t_body = self.transform_body(img_body)
        t_face = self.transform_face(img_face)
        
        # Unsqueeze to add Temporal dimension (T=1)
        # Shape: (1, 3, 224, 224)
        t_body = t_body.unsqueeze(0)
        t_face = t_face.unsqueeze(0)
        
        # Label is assumed to be 0-based in the file (0..6)
        label_idx = label
        
        return t_face, t_body, label_idx

    def __len__(self):
        return len(self.samples)

def caers_train_data_loader(root_dir, list_file, image_size, batch_size, num_workers=4, bounding_box_json=None):
    dataset = CAERSDataset(root_dir, list_file, mode='train', image_size=image_size, bounding_box_json=bounding_box_json)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

def caers_val_data_loader(root_dir, list_file, image_size, batch_size, num_workers=4, bounding_box_json=None):
    dataset = CAERSDataset(root_dir, list_file, mode='val', image_size=image_size, bounding_box_json=bounding_box_json)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def caers_test_data_loader(root_dir, list_file, image_size, batch_size, num_workers=4, bounding_box_json=None):
    dataset = CAERSDataset(root_dir, list_file, mode='test', image_size=image_size, bounding_box_json=bounding_box_json)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
