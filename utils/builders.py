# builders.py

import argparse
from typing import Tuple
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from dataloader.caer_s_dataloader import caer_s_data_loader
from dataloader.caer_video_dataloader import caer_video_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import get_class_counts


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    # [LUỒNG 3.1: LOAD BACKBONE]
    # Tải pretrained CLIP (ViT-B/16)
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')

    print("\nInput Text Prompts:")
    # Handle the case where input_text is a list of lists for prompt ensembling
    if any(isinstance(i, list) for i in input_text):
        for class_prompts in input_text:
            print(f"- Class: {class_prompts}")
    else:
        for text in input_text:
            print(text)


    print("\nInstantiating GenerateModel...")
    # [LUỒNG 3.2: INIT ARCHITECTURE]
    # Khởi tạo kiến trúc tổng thể (Adapter, Prompt Learner, Temporal)
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze CLIP image encoder if lr_image_encoder is 0
    # Otherwise, make it trainable.
    if args.lr_image_encoder > 0:
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = True

    # [LUỒNG 3.3: SET TRAINABLE PARAMS]
    # Chỉ train các module phụ trợ (Adapter, Prompt Learner, Temporal)
    trainable_params_keywords = ["temporal_net", "prompt_learner", "temporal_net_body", "project_fc", "face_adapter"]
    
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    根据数据集和文本类型获取 class_names 和 input_text（用于生成 CLIP 模型文本输入）。

    Returns:
        class_names: 类别名称，用于混淆矩阵等
        input_text: 输入文本，用于传入模型
    """
    if args.dataset == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction.']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
        ensemble_prompts = prompt_ensemble_5
    elif args.dataset == "CAER-S":
        class_names = class_names_7
        class_names_with_context = class_names_7 # Fallback to names if no context
        class_descriptor = class_names_7 # Fallback
        ensemble_prompts = prompt_ensemble_7
    elif args.dataset == "CAER" or args.dataset == "CAER_Combined":
        class_names = class_names_7
        class_names_with_context = class_names_7
        class_descriptor = class_names_7
        ensemble_prompts = prompt_ensemble_7
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    elif args.text_type == "prompt_ensemble":
        input_text = ensemble_prompts
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    # [LUỒNG 4.1: DATA CONFIG]
    train_annotation_file_path = args.train_annotation
    val_annotation_file_path = args.val_annotation
    test_annotation_file_path = args.test_annotation
    
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    # [LUỒNG 4.2: DATASETS]
    # Khởi tạo Dataset object (đọc video, transform)
    if args.dataset == "CAER-S":
        print("Loading CAER-S train data...")
        train_data = caer_s_data_loader(
            root_dir=args.root_dir, mode='train', num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )
        print(f"Total number of training images: {len(train_data)}")
        
        print("Loading CAER-S validation data...")
        val_data = caer_s_data_loader(
            root_dir=args.root_dir, mode='val', num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )

        print("Loading CAER-S test data...")
        # CAER-S only has train and test
        test_data = caer_s_data_loader(
            root_dir=args.root_dir, mode='test', num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )

    elif args.dataset == "CAER":
        print("Loading CAER video train data...")
        train_data = caer_video_data_loader(
            root_dir=args.root_dir, mode='train', num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )
        print(f"Total number of training videos: {len(train_data)}")
        
        print("Loading CAER video validation data...")
        val_data = caer_video_data_loader(
            root_dir=args.root_dir, mode='val', num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )

        print("Loading CAER video test data...")
        test_data = caer_video_data_loader(
            root_dir=args.root_dir, mode='test', num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )

    elif args.dataset == "CAER_Combined":
        print("Loading Combined CAER-S + CAER datasets...")
        
        # CAER-S
        train_s = caer_s_data_loader(root_dir="CAER-S", mode='train', num_segments=args.num_segments, duration=args.duration, image_size=args.image_size)
        val_s = caer_s_data_loader(root_dir="CAER-S", mode='val', num_segments=args.num_segments, duration=args.duration, image_size=args.image_size)
        test_s = caer_s_data_loader(root_dir="CAER-S", mode='test', num_segments=args.num_segments, duration=args.duration, image_size=args.image_size)
        
        # CAER
        train_v = caer_video_data_loader(root_dir="CAER", mode='train', num_segments=args.num_segments, duration=args.duration, image_size=args.image_size)
        val_v = caer_video_data_loader(root_dir="CAER", mode='val', num_segments=args.num_segments, duration=args.duration, image_size=args.image_size)
        test_v = caer_video_data_loader(root_dir="CAER", mode='test', num_segments=args.num_segments, duration=args.duration, image_size=args.image_size)
        
        # Combine
        train_data = torch.utils.data.ConcatDataset([train_s, train_v])
        val_data = torch.utils.data.ConcatDataset([val_s, val_v])
        test_data = torch.utils.data.ConcatDataset([test_s, test_v])
        
        print(f"Total Combined Training Samples: {len(train_data)} (Img: {len(train_s)} + Vid: {len(train_v)})")

    else:
        print("Loading RAER train data...")
        train_data = train_data_loader(
            root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
            bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body,
            num_classes=num_classes
        )
        print(f"Total number of training images: {len(train_data)}")
        
        print("Loading RAER validation data...")
        val_data = test_data_loader(
            root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body,
            num_classes=num_classes
        )

        print("Loading RAER test data...")
        test_data = test_data_loader(
            root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body,
            num_classes=num_classes
        )

    print("Creating DataLoader instances...")
    
    sampler = None
    shuffle = True
    # Disable WeightedRandomSampler for CAER/CAER-S/CAER_Combined as they don't use annotation files
    if args.use_weighted_sampler and args.dataset not in ["CAER-S", "CAER", "CAER_Combined"]:
        print("=> Using WeightedRandomSampler.")
        class_counts = get_class_counts(train_annotation_file_path)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        
        # Create a weight for each sample
        sample_weights = []
        with open(train_annotation_file_path, 'r') as f:
            for line in f:
                label = int(line.strip().split()[2]) -1 # label is 1-based
                sample_weights.append(class_weights[label])
        
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False # Sampler and shuffle are mutually exclusive

    # [LUỒNG 4.3: DATALOADERS]
    # Đóng gói Dataset vào DataLoader để batching
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader