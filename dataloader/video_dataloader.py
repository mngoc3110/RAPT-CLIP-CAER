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

# =========================
# OPTIONAL IMPORT (DAiSEE)
# =========================
try:
    from dataloader.daisee_dataloader import daisee_train_data_loader, daisee_test_data_loader
except Exception:
    daisee_train_data_loader = None
    daisee_test_data_loader = None


# Custom Transform for List of Images (Group Transform)
class GroupRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            return [img.convert('L').convert('RGB') for img in img_group]
        return img_group


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    """
    Supports:
      - Video folders (frames)
      - Video files (.avi, .mp4, ...)
      - Single image files (CAER-S .png/.jpg)
    Annotation supports:
      - 2 columns:  path label
      - 3 columns:  path num_frames label
      - path with spaces (rare): handled
    BBox supports:
      - flat dict:  key -> [x1,y1,x2,y2]    (your /kaggle/working/bboxes/*.json)
      - nested dict: key -> {frame.jpg: [..]} (old video style)
    """

    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        list_file,
        num_segments,
        duration,
        mode,
        transform,
        image_size,
        bounding_box_face,
        bounding_box_body,
        crop_body=False,
        root_dir="",
        num_classes=8,
        label_shift=0,           # IMPORTANT: CAER-S labels are 0..6 => label_shift=0
        bbox_prefix="CAER",      # IMPORTANT: your bbox keys are CAER/<rel_no_ext>
        balance_data=False,      # Internal oversampling
    ):
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
        self.balance_data = balance_data

        self.label_shift = int(label_shift)
        self.bbox_prefix = bbox_prefix

        self._read_sample()
        self._parse_list()
        self._read_boxs()
        if self.crop_body:
            self._read_body_boxes()

    # ----------------------------
    # BBOX LOAD
    # ----------------------------
    def _read_boxs(self):
        with open(self.bounding_box_face, "r") as f:
            self.boxs = json.load(f)

    def _read_body_boxes(self):
        if self.bounding_box_body:
            with open(self.bounding_box_body, "r") as f:
                self.body_boxes = json.load(f)

    # ----------------------------
    # HELPERS
    # ----------------------------
    def _cv2pil(self, im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype("uint8"))
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
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return im, r

    def _face_detect(self, img, box, margin, mode="face"):
        # Fallback: return original image if no box
        if box is None:
            return img

        left, upper, right, lower = box
        left = int(left); upper = int(upper); right = int(right); lower = int(lower)

        left = max(0, left - margin)
        upper = max(0, upper - margin)
        right = min(img.width, right + margin)
        lower = min(img.height, lower + margin)

        if right <= left or lower <= upper:
            return img

        if mode == "face":
            return img.crop((left, upper, right, lower))
        elif mode == "body":
            occluded_image = img.copy()
            draw = ImageDraw.Draw(occluded_image)
            draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
            return occluded_image
        return img

    def _rel_from_abs(self, abs_path: str) -> str:
        """Make record.path (absolute) -> relative to root_dir, normalized with /."""
        p = abs_path.replace("\\", "/")
        root = (self.root_dir or "").replace("\\", "/").rstrip("/")
        if root and p.startswith(root + "/"):
            p = p[len(root) + 1:]
        return p.lstrip("./")

    def _bbox_key_from_record(self, record_path_abs: str) -> str:
        """Build bbox key like: CAER/train/train/Angry/0566"""
        rel = self._rel_from_abs(record_path_abs)
        rel_no_ext = os.path.splitext(rel)[0]
        return f"{self.bbox_prefix}/{rel_no_ext}".replace("\\", "/")

    def _lookup_box(self, bbox_dict, key: str, frame_key: str = None):
        """
        Supports:
          - flat: bbox_dict[key] = [x1,y1,x2,y2]
          - nested: bbox_dict[key][frame_key] = [...]
        """
        if bbox_dict is None:
            return None
        if key not in bbox_dict:
            return None
        v = bbox_dict[key]
        if isinstance(v, list) and len(v) == 4:
            return v
        if isinstance(v, dict) and frame_key is not None and frame_key in v:
            return v[frame_key]
        return None

    # ----------------------------
    # READ ANNOTATION
    # ----------------------------
    def _read_sample(self):
        self.sample_list = []
        with open(self.list_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()  # split on any whitespace
                # support:
                #   path label
                #   path num_frames label
                #   path(with spaces) num_frames label
                if len(parts) == 2:
                    path = parts[0]
                    num_frames = "1"
                    label = parts[1]
                elif len(parts) >= 3:
                    # last two are num_frames + label OR something + label
                    # In our pipeline, we will treat: ... <num_frames> <label>
                    path = " ".join(parts[:-2])
                    num_frames = parts[-2]
                    label = parts[-1]
                else:
                    # malformed
                    continue

                self.sample_list.append([path, num_frames, label])

        # Internal Oversampling for Minority Classes (if mode is train)
        if self.mode == "train" and self.balance_data:
            print("=> Applying internal oversampling to balance dataset...")
            labels = [int(x[2]) for x in self.sample_list]
            unique_labels = sorted(list(set(labels)))
            label_counts = {lbl: labels.count(lbl) for lbl in unique_labels}
            max_count = max(label_counts.values())
            
            balanced_list = []
            for lbl in unique_labels:
                samples_of_label = [x for x in self.sample_list if int(x[2]) == lbl]
                count = len(samples_of_label)
                if count == 0: continue
                # Multiply samples to reach close to max_count (capped to avoid massive dataset)
                multiplier = min(int(max_count / count), 4) # cap at 4x increase
                balanced_list.extend(samples_of_label * multiplier)
            
            self.sample_list = balanced_list
            random.shuffle(self.sample_list)
            print(f"=> Dataset balanced. New size: {len(self.sample_list)}")

    def _parse_list(self):
        self.video_list = [
            VideoRecord([os.path.join(self.root_dir, item[0])] + item[1:])
            for item in self.sample_list
        ]
        print(("video number:%d" % (len(self.video_list))))

    # ----------------------------
    # INDICES
    # ----------------------------
    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), "edge")
        return offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), "edge")
        return offsets

    # ----------------------------
    # DATASET API
    # ----------------------------
    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == "train":
            segment_indices = self._get_train_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        # Detect type:
        # - directory of frames
        # - single image file (CAER-S)
        # - video file
        abs_path = record.path
        ext = os.path.splitext(abs_path)[1].lower()

        is_dir_frames = os.path.isdir(abs_path)
        is_image_file = os.path.isfile(abs_path) and (ext in self.IMG_EXT)
        is_video_file = (not is_dir_frames) and (not is_image_file)

        # Load frame list / frame count
        cap = None
        if is_dir_frames:
            video_frames_path = glob.glob(os.path.join(abs_path, "*"))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
        elif is_image_file:
            video_frames_path = [abs_path]  # treat as 1-frame "video"
            num_real_frames = 1
        else:
            cap = cv2.VideoCapture(abs_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {abs_path}, returning zeros.")
                num_real_frames = 0
            else:
                num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_real_frames == 0:
            dummy_shape = (self.num_segments * self.duration, 3, self.image_size, self.image_size)
            if cap is not None:
                cap.release()
            # IMPORTANT: do NOT shift to -1
            return torch.zeros(dummy_shape), torch.zeros(dummy_shape), int(record.label) + self.label_shift

        indices = np.clip(indices, 0, num_real_frames - 1)

        images_body = []
        images_face = []

        bbox_key = self._bbox_key_from_record(record.path)  # CAER/<rel_no_ext>
        frame_key = "0.jpg"  # for nested dict format, if needed

        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.duration):
                # 1) Read image frame
                if is_dir_frames:
                    img_path = video_frames_path[p]
                    try:
                        img_pil = Image.open(img_path).convert("RGB")
                    except Exception:
                        img_pil = Image.new("RGB", (self.image_size, self.image_size))
                elif is_image_file:
                    try:
                        img_pil = Image.open(video_frames_path[0]).convert("RGB")
                    except Exception:
                        img_pil = Image.new("RGB", (self.image_size, self.image_size))
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                    ret, frame = cap.read()
                    if ret:
                        img_cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_cv_rgb)
                    else:
                        img_pil = Image.new("RGB", (self.image_size, self.image_size))

                # 2) Lookup bbox (flat or nested)
                box_face = self._lookup_box(self.boxs, bbox_key, frame_key)
                img_pil_face = self._face_detect(img_pil, box_face, margin=10, mode="face")

                # 3) Body crop optional
                img_pil_body = img_pil
                if self.crop_body and hasattr(self, "body_boxes"):
                    box_body = self._lookup_box(self.body_boxes, bbox_key, frame_key)
                    if box_body is not None:
                        left, upper, right, lower = box_body
                        left = max(0, int(left)); upper = max(0, int(upper))
                        right = min(img_pil.width, int(right)); lower = min(img_pil.height, int(lower))
                        if right > left and lower > upper:
                            img_pil_body = img_pil.crop((left, upper, right, lower))

                # resize body (face is resized via transforms)
                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, _ = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)

                images_body.append(img_pil_body)
                images_face.append(img_pil_face)

                if p < num_real_frames - 1:
                    p += 1

        if cap is not None:
            cap.release()

        process_body = self.transform(images_body)       # (C*T, H, W)
        process_face = self.transform(images_face)       # (C*T, H, W)

        process_body = process_body.view(-1, 3, self.image_size, self.image_size)
        process_face = process_face.view(-1, 3, self.image_size, self.image_size)

        # IMPORTANT: CAER-S labels are already 0..6 => no "-1"
        target = int(record.label) + self.label_shift
        return process_face, process_body, target

    def __len__(self):
        return len(self.video_list)


# =========================
# PUBLIC API
# =========================
def train_data_loader(root_dir, list_file, num_segments, duration, image_size, dataset_name,
                      bounding_box_face, bounding_box_body, crop_body=False, num_classes=8):

    if dataset_name == "DAiSEE":
        if daisee_train_data_loader is None:
            raise ImportError("DAiSEE loader not available, but dataset_name=DAiSEE was requested.")
        print(f"=> Using DAiSEE smart dataloader...")
        return daisee_train_data_loader(
            root_dir, list_file, num_segments, duration, image_size,
            bounding_box_face, bounding_box_body, crop_body, num_classes
        )

    if dataset_name in ["RAER", "CAER", "CAER-S"]:
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            GroupRandomGrayscale(p=0.2),
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
        ])
        # CAER-S labels are 0..6 => shift 0
        label_shift = 0
        bbox_prefix = "CAER"
    else:
        train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
        ])
        label_shift = 0
        bbox_prefix = "CAER"

    train_data = VideoDataset(
        root_dir=root_dir,
        list_file=list_file,
        num_segments=num_segments,
        duration=duration,
        mode="train",
        transform=train_transforms,
        image_size=image_size,
        bounding_box_face=bounding_box_face,
        bounding_box_body=bounding_box_body,
        crop_body=crop_body,
        num_classes=num_classes,
        label_shift=label_shift,
        bbox_prefix=bbox_prefix,
    )
    return train_data


def test_data_loader(root_dir, list_file, num_segments, duration, image_size,
                     bounding_box_face, bounding_box_body, crop_body=False, num_classes=8):

    test_transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor(),
    ])

    test_data = VideoDataset(
        root_dir=root_dir,
        list_file=list_file,
        num_segments=num_segments,
        duration=duration,
        mode="test",
        transform=test_transform,
        image_size=image_size,
        bounding_box_face=bounding_box_face,
        bounding_box_body=bounding_box_body,
        crop_body=crop_body,
        num_classes=num_classes,
        label_shift=0,
        bbox_prefix="CAER",
    )
    return test_data