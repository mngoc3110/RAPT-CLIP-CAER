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
from dataloader.daisee_dataloader import daisee_train_data_loader, daisee_test_data_loader

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        return int(self._data[1])

    @property       # 标签
    def label(self):
        return int(self._data[2])

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size,bounding_box_face,bounding_box_body, crop_body=False, root_dir="", num_classes=8):
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
        
        # Debugging: Initialize for saving sample images
        self.debug_samples_path = 'debug_samples'
        os.makedirs(self.debug_samples_path, exist_ok=True)
        self._saved_samples = {i: 0 for i in range(num_classes)}
        
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        if self.crop_body: # Only read body boxes if cropping is enabled
            self._read_body_boxes()

    def _read_boxs(self):
        with open(self.bounding_box_face, 'r') as f:
            self.boxs = json.load(f)


    
    def _read_body_boxes(self):
        if self.bounding_box_body:
            with open(self.bounding_box_body, 'r') as f:
                self.body_boxes = json.load(f)


    def _cv2pil(self,im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self,im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self,im, width, height):
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

    def _face_detect(self,img,box,margin,mode = 'face'):
        if box is None:
            if mode == 'face':
                # Return a black image of the same size if face is not detected
                return Image.new('RGB', img.size, (0, 0, 0))
            return img
        else:
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
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
    
    def _read_sample(self):
        # tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # self.sample_list = [item for item in tmp]
        
        self.sample_list = []
        with open(self.list_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) > 3:
                    # Path contains spaces, join all parts except the last two
                    path = ' '.join(parts[:-2])
                    num_frames = parts[-2]
                    label = parts[-1]
                    self.sample_list.append([path, num_frames, label])
                else:
                    self.sample_list.append(parts)


    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, class_idx]
        #
        self.video_list = [VideoRecord([os.path.join(self.root_dir, item[0])] + item[1:]) for item in self.sample_list]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in each part randomly
        #
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in the mid of each part
        #
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
        # Check if record.path is a directory (frames) or a file (video)
        if os.path.isdir(record.path):
            video_frames_path = glob.glob(os.path.join(record.path, '*'))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
            is_video_file = False
        else:
            # Assume it's a video file
            is_video_file = True
            cap = cv2.VideoCapture(record.path)
            num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Don't release cap yet, we need it to read frames
            if not cap.isOpened():
                 print(f"Warning: Could not open video file {record.path}, returning zeros.")
                 num_real_frames = 0

        if num_real_frames == 0:
            print(f"Warning: No frames found for video {record.path}, returning zeros.")
            dummy_shape = (self.num_segments * self.duration, 3, self.image_size, self.image_size)
            if is_video_file and 'cap' in locals(): cap.release()
            return torch.zeros(dummy_shape), torch.zeros(dummy_shape), record.label - 1

        # Clamp indices to be valid
        indices = np.clip(indices, 0, num_real_frames - 1)
        
        images = list()
        images_face = list()
        
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                # Initialize variables to avoid UnboundLocalError
                parent_dir = ""
                video_key = ""
                frame_key = ""
                box = None
                img_pil = None # Ensure img_pil is initialized

                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                    ret, frame = cap.read()
                    if ret:
                        # cv2 reads in BGR, convert to RGB for PIL
                        img_cv = frame # Keep BGR for _face_detect or convert? 
                        # _face_detect uses PIL. Let's convert to PIL.
                        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_cv_rgb)
                        
                        # CAER specific BBox Lookup Logic
                        # Path in record: ./CAER/train/Anger/0001.avi
                        # Key in JSON: CAER/train/Anger/0001
                        # Subkey in JSON: 0.jpg (frame index)
                        
                        # Mock path for box lookup (parent dir and filename)
                        # For video file, parent is dir, filename is video name + frame index?
                        # This bbox logic might need adjustment for video files if you have frame-level boxes.
                        # For now, let's assume we use the video path or parent dir for box lookup key.
                        parent_dir = os.path.dirname(record.path)
                        file_name = os.path.basename(record.path) # Use video filename

                        # Remove './' if present
                        rel_path = record.path.replace('./', '')
                        # Remove extension
                        video_key = os.path.splitext(rel_path)[0]
                        
                        # Robust key lookup: 
                        # If the full path key isn't found, try stripping top-level directories
                        # e.g., 'dataset/RAER/images/...' -> 'RAER/images/...'
                        if video_key not in self.boxs:
                            parts = video_key.split(os.sep)
                            # Try progressively shorter suffixes
                            for i in range(1, len(parts)):
                                sub_key = '/'.join(parts[i:]) # JSON keys usually use forward slashes
                                if sub_key in self.boxs:
                                    video_key = sub_key
                                    break
                        
                        frame_key = f"{p}.jpg" # CAER uses 0.jpg, 1.jpg... based on frame index
                        
                        # Debug warning if face box still not found
                        if video_key not in self.boxs and i == 0: # Print once per clip
                             # print(f"Warning: Video key '{video_key}' (derived from '{record.path}') not found in face bounding boxes.")
                             pass # Suppress for now to avoid spam, but logic above should fix it.
                    else:
                        # Failed to read frame, use black image
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))
                        # parent_dir is "" (initialized above)
                else:
                    img_path = video_frames_path[p]
                    parent_dir = os.path.dirname(img_path)
                    file_name = os.path.basename(img_path)
                    try:
                        img_pil = Image.open(img_path)
                    except:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))
                    
                    # Logic for frame folders if needed...
                    # For now assume generic lookup works or implement similar key logic if frame folders follow CAER structure
                    
                
                # Try CAER lookup style first
                if video_key and video_key in self.boxs:
                    if frame_key in self.boxs[video_key]:
                        box = self.boxs[video_key][frame_key]
                
                # Fallback to generic lookup if not found (parent dir -> filename)
                if box is None and parent_dir:
                    if parent_dir in self.boxs:
                         if file_name in self.boxs[parent_dir]:
                             box = self.boxs[parent_dir][file_name]

                # Perform face detection / cropping
                if img_pil is None: # Safety check
                     img_pil = Image.new('RGB', (self.image_size, self.image_size))
                     
                img_pil_face = self._face_detect(img_pil, box, margin=20, mode='face')

                # Debugging: Save sample images
                current_label = record.label - 1
                if self.mode == 'train' and self._saved_samples.get(current_label, 0) < 5:
                    sample_path = os.path.join(self.debug_samples_path, f'class_{current_label}')
                    os.makedirs(sample_path, exist_ok=True)
                    # Unique name
                    base_name = os.path.basename(record.path)
                    img_name = f'sample_{self._saved_samples[current_label]}_{base_name}_{p}.jpg'
                    img_pil_face.save(os.path.join(sample_path, img_name))
                    self._saved_samples[current_label] += 1

                if self.crop_body:
                    body_box = None
                    # Try CAER lookup style first
                    if video_key and video_key in self.body_boxes:
                        if frame_key in self.body_boxes[video_key]:
                            body_box = self.body_boxes[video_key][frame_key]
                    
                    # Fallback to generic lookup if not found
                    if body_box is None and parent_dir:
                        body_box_path = parent_dir
                        if body_box_path in self.body_boxes:
                            body_box = self.body_boxes[body_box_path]

                    if body_box is not None:
                        left, upper, right, lower = body_box
                        img_pil_body = img_pil.crop((left, upper, right, lower))
                    else:
                        img_pil_body = img_pil
                else:
                    img_pil_body = img_pil # Use full frame if not cropping body

                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, r = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                seg_imgs = [img_pil_body]
                
                seg_imgs_face = [img_pil_face]

                images.extend(seg_imgs)
                images_face.extend(seg_imgs_face)
                if p < num_real_frames - 1:
                    p += 1
        
        if is_video_file:
            cap.release()

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        images_face = self.transform(images_face)
        images_face = torch.reshape(images_face, (-1, 3, self.image_size, self.image_size))
        return images_face,images,record.label-1

    def __len__(self):
        return len(self.video_list)


def train_data_loader(root_dir, list_file, num_segments, duration, image_size,dataset_name,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8):
    if dataset_name == 'DAiSEE':
        print(f"=> Using DAiSEE smart dataloader...")
        return daisee_train_data_loader(root_dir, list_file, num_segments, duration, image_size, 
                                        bounding_box_face, bounding_box_body, crop_body, num_classes)
        
    if dataset_name == "RAER" or dataset_name == "CAER":
         train_transforms = torchvision.transforms.Compose([
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    else:
         # Default transforms for other datasets like CK+
         train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
            
    
    train_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                              num_segments=num_segments, #16
                              duration=duration, #1
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body,
                              crop_body=crop_body,
                              num_classes=num_classes
                              )
    return train_data


def test_data_loader(root_dir, list_file, num_segments, duration, image_size,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8):
    # We don't get dataset_name here usually, but if we did we could dispatch.
    # However, test_data_loader signature in main.py call might not pass dataset_name?
    # Let's check main.py call site.
    # But wait, main.py doesn't pass dataset_name to test_data_loader in standard calls usually.
    # Let's see if we can infer it or if we should add it.
    pass 
    
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    test_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body,
                             crop_body=crop_body,
                             num_classes=num_classes
                             )
    return test_data