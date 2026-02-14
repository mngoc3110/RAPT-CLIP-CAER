import os
import glob
import json
import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils import data
from numpy.random import randint
from dataloader.video_transform import *

class DAiSEERecord(object):
    def __init__(self, row, root_dir):
        self._data = row
        self._root_dir = root_dir
        self._resolved_path = None # Cache for the actual path

    @property
    def clip_id(self):
        # row[0] is the relative path from annotation file, e.g. "Train/110001/1100011002/frames" 
        # or just "1100011002" depending on how we generated it.
        # Let's assume the annotation file contains relative paths.
        return self._data[0]

    @property
    def path(self):
        if self._resolved_path:
            return self._resolved_path
            
        # Strategy: The annotation file provides a relative path.
        # We need to check if it exists relative to root_dir.
        # If not, we might need to search for the clip ID.
        
        rel_path = self._data[0]
        full_path = os.path.join(self._root_dir, rel_path)
        
        if os.path.exists(full_path):
            self._resolved_path = full_path
            return full_path
            
        # Fallback: If path is not found directly, maybe it's a video file missing extension?
        # Or maybe the annotation path was "clean" but file has extension.
        if not os.path.exists(full_path):
             for ext in ['.avi', '.mp4', '.mov']:
                 if os.path.exists(full_path + ext):
                     self._resolved_path = full_path + ext
                     return self._resolved_path
        
        # If still not found, return the full_path anyway (dataloader will handle missing file)
        return full_path

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class DAiSEEDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size, bounding_box_face, bounding_box_body, crop_body=False, root_dir="", num_classes=4):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self.crop_body = crop_body
        self.root_dir = root_dir
        
        self.debug_samples_path = 'debug_samples'
        os.makedirs(self.debug_samples_path, exist_ok=True)
        self._saved_samples = {i: 0 for i in range(num_classes)}
        
        self._parse_list()
        
        # We skip reading boxes for now as DAiSEE usually relies on full frame or specific crops
        # But to keep compatibility, we init empty boxes if file fails
        try:
            with open(self.bounding_box_face, 'r') as f:
                self.boxs = json.load(f)
        except:
            self.boxs = {}
            
        try:
            if self.crop_body:
                with open(self.bounding_box_body, 'r') as f:
                    self.body_boxes = json.load(f)
            else:
                self.body_boxes = {}
        except:
            self.body_boxes = {}

    def _cv2pil(self, im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self, im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self, im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self, img, box, margin, mode='face'):
        if box is None:
            return img
        else:
            left, upper, right, lower = box
            left = int(left); upper = int(upper); right = int(right); lower = int(lower)
            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)
            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image

    def _parse_list(self):
        self.video_list = []
        with open(self.list_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 3:
                    # Handle paths with spaces if any
                    path = ' '.join(parts[:-2])
                    num_frames = parts[-2]
                    label = parts[-1]
                    self.video_list.append(DAiSEERecord([path, num_frames, label], self.root_dir))
        
        print(f'DAiSEE {self.mode} samples: {len(self.video_list)}')

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        path = record.path
        is_video_file = os.path.isfile(path) and path.lower().endswith(('.avi', '.mp4', '.mov'))
        
        video_frames_path = []
        num_real_frames = 0
        cap = None

        if is_video_file:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                print(f"Warning: Could not open video file {path}")
        elif os.path.isdir(path):
            video_frames_path = glob.glob(os.path.join(path, '*'))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
        
        # If frame count mismatch or file not found
        if num_real_frames == 0:
            # print(f"Warning: No frames/video found for {path}, returning zeros.")
            dummy_shape = (self.num_segments * self.duration, 3, self.image_size, self.image_size)
            if cap: cap.release()
            return torch.zeros(dummy_shape), torch.zeros(dummy_shape), record.label - 1

        indices = np.clip(indices, 0, num_real_frames - 1)
        
        images = []
        images_face = []
        
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_pil = None
                
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                    ret, frame = cap.read()
                    if ret:
                        img_cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_cv_rgb)
                else:
                    # Folder of frames
                    if p < len(video_frames_path):
                        try:
                            img_pil = Image.open(video_frames_path[p]).convert('RGB')
                        except:
                            pass

                if img_pil is None:
                    img_pil = Image.new('RGB', (self.image_size, self.image_size))

                # DAiSEE specific: Center Crop if no bbox (Dummy box fallback)
                # Most DAiSEE videos have the subject in the center.
                # A 60-70% center crop helps remove background noise.
                if self.bounding_box_face == "" or "dummy" in self.bounding_box_face:
                     w, h = img_pil.size
                     # Crop scale 0.7 seems reasonable for webcam footage
                     crop_scale = 0.7 
                     crop_w = w * crop_scale
                     crop_h = h * crop_scale
                     
                     # Ensure we don't crop too small
                     if crop_w < self.image_size: crop_w = w
                     if crop_h < self.image_size: crop_h = h
                     
                     left = (w - crop_w) / 2
                     top = (h - crop_h) / 2
                     img_pil_face = img_pil.crop((left, top, left + crop_w, top + crop_h))
                else:
                     img_pil_face = self._face_detect(img_pil, None, margin=0, mode='face')

                # Body stream uses the same image if no cropping
                img_pil_body = img_pil

                # Resize
                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, _ = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                
                images.append(img_pil_body)
                images_face.append(img_pil_face)
                
                if p < num_real_frames - 1:
                    p += 1
        
        if cap:
            cap.release()

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        images_face = self.transform(images_face)
        images_face = torch.reshape(images_face, (-1, 3, self.image_size, self.image_size))
        
        return images_face, images, record.label - 1

    def __len__(self):
        return len(self.video_list)

def daisee_train_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    train_transforms = torchvision.transforms.Compose([
        RandomRotation(4),
        GroupResize(image_size),
        GroupRandomHorizontalFlip(),
        Stack(),
        ToTorchFormatTensor()])
    
    return DAiSEEDataset(root_dir=root_dir, list_file=list_file,
                         num_segments=num_segments,
                         duration=duration,
                         mode='train',
                         transform=train_transforms,
                         image_size=image_size,
                         bounding_box_face=bounding_box_face,
                         bounding_box_body=bounding_box_body,
                         crop_body=crop_body,
                         num_classes=num_classes)

def daisee_test_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=4):
    test_transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor()])
    
    return DAiSEEDataset(root_dir=root_dir, list_file=list_file,
                         num_segments=num_segments,
                         duration=duration,
                         mode='test',
                         transform=test_transform,
                         image_size=image_size,
                         bounding_box_face=bounding_box_face,
                         bounding_box_body=bounding_box_body,
                         crop_body=crop_body,
                         num_classes=num_classes)
