import os
import glob
import json
import random
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils import data
from numpy.random import randint
from dataloader.video_transform import *

class CKPlusDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size, root_dir="", num_classes=7):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.root_dir = root_dir
        
        # Read samples (each line is now a path to a single IMAGE)
        # Format: path num_frames label_id
        self.samples = []
        with open(self.list_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                if len(parts) > 3:
                    # Path contains spaces
                    path = ' '.join(parts[:-2])
                    num_frames = parts[-2]
                    label_id = parts[-1]
                    self.samples.append([path, num_frames, label_id])
                else:
                    self.samples.append(parts)
        print(f'CK+ {mode} samples (images): {len(self.samples)}')

    def __getitem__(self, index):
        img_path, _, label_id = self.samples[index]
        label_id = int(label_id)
        
        # Load single image
        full_path = os.path.join(self.root_dir, img_path)
        try:
            img_pil = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {full_path} - {e}")
            # Return dummy image if failed
            img_pil = Image.new('RGB', (self.image_size, self.image_size))

        # Since the model expects a "video" (sequence of frames), 
        # we provide the same image repeated 'num_segments' times.
        # This allows the temporal transformer to work even with single images.
        images = [img_pil] * self.num_segments
        
        processed_images = self.transform(images)
        processed_images = torch.reshape(processed_images, (-1, 3, self.image_size, self.image_size))
        
        # Return same image for face and body streams
        return processed_images, processed_images, label_id - 1

    def __len__(self):
        return len(self.samples)

def ckplus_train_data_loader(root_dir, list_file, num_segments, duration, image_size):
    # Mimicking TensorFlow's ImageDataGenerator with torchvision
    train_transforms = torchvision.transforms.Compose([
        GroupResize(image_size),
        # Random Rotation, Shift, Shear, Zoom (Affine)
        GroupRandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
        GroupRandomHorizontalFlip(),
        Stack(),
        ToTorchFormatTensor()
    ])
    
    return CKPlusDataset(root_dir=root_dir, list_file=list_file,
                         num_segments=num_segments,
                         duration=duration,
                         mode='train',
                         transform=train_transforms,
                         image_size=image_size)

def ckplus_test_data_loader(root_dir, list_file, num_segments, duration, image_size):
    test_transforms = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor()
    ])
    
    return CKPlusDataset(root_dir=root_dir, list_file=list_file,
                         num_segments=num_segments,
                         duration=duration,
                         mode='test',
                         transform=test_transforms,
                         image_size=image_size)

# Helper Class for Group Affine (matching TF behavior)
class GroupRandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.worker = transforms.RandomAffine(degrees, translate, scale, shear)

    def __call__(self, img_group):
        # We must apply the SAME random transformation to all frames in the "video"
        # to maintain temporal consistency
        params = self.worker.get_params(self.worker.degrees, self.worker.translate, self.worker.scale, self.worker.shear, img_group[0].size)
        return [transforms.functional.affine(img, *params) for img in img_group]
