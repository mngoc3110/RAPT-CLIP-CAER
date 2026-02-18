# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from dataloader.ckplus_dataloader import ckplus_train_data_loader, ckplus_test_data_loader
from dataloader.daisee_dataloader import daisee_train_data_loader, daisee_test_data_loader
from dataloader.caer_s_dataloader import CAERSDataset
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
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
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze CLIP image encoder if lr_image_encoder is 0
    # Otherwise, make it trainable.
    if args.lr_image_encoder > 0:
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = True

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
    dataset_name = args.dataset.strip()
    print(f"DEBUG: get_class_info processing dataset '{dataset_name}'")

    if dataset_name == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
        ensemble_prompts = prompt_ensemble_5
    elif dataset_name == "CK+":
        class_names = class_names_ckplus
        class_names_with_context = class_names_with_context_ckplus
        class_descriptor = class_descriptor_ckplus
        ensemble_prompts = prompt_ensemble_ckplus
    elif dataset_name == "SFER":
        class_names = class_names_sfer
        class_names_with_context = class_names_with_context_sfer
        class_descriptor = class_descriptor_sfer
        ensemble_prompts = prompt_ensemble_sfer
    elif dataset_name == "DAiSEE":
        class_names = class_names_daisee
        class_names_with_context = class_names_with_context_daisee
        class_descriptor = class_descriptor_daisee
        ensemble_prompts = prompt_ensemble_daisee
    elif dataset_name == "CAER" or dataset_name == "CAER-S":
        class_names = class_names_caer
        class_names_with_context = class_names_with_context_caer
        class_descriptor = class_descriptor_caer
        ensemble_prompts = prompt_ensemble_caer
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented yet.")

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
    train_annotation_file_path = args.train_annotation
    val_annotation_file_path = args.val_annotation
    test_annotation_file_path = args.test_annotation
    
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    # Debug print
    print(f"DEBUG: args.dataset = '{args.dataset}'")

    if args.dataset.strip() == "CK+" or args.dataset.strip() == "SFER":
        print(f"=> Using {args.dataset} specific dataloader...")
        train_data = ckplus_train_data_loader(
            root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )
        val_data = ckplus_test_data_loader(
            root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )
        test_data = ckplus_test_data_loader(
            root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size
        )
    elif args.dataset.strip() == "DAiSEE":
        print(f"=> Using DAiSEE smart dataloader...")
        train_data = daisee_train_data_loader(
            root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body, num_classes=num_classes
        )
        val_data = daisee_test_data_loader(
            root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body, num_classes=num_classes
        )
        test_data = daisee_test_data_loader(
            root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body, num_classes=num_classes
        )
    elif args.dataset.strip() == "CAER-S":
        print(f"=> Using CAER-S specific dataloader...")
        # CAER-S now uses standard annotation files generated by preprocess_caer.py
        
        # Pass args.bounding_box_face as the JSON file for faces
        bbox_json = args.bounding_box_face
        print(f"=> Face Bounding Box JSON: {bbox_json}")

        train_loader = caers_train_data_loader(
            root_dir=args.root_dir, 
            list_file=train_annotation_file_path, 
            image_size=args.image_size, 
            batch_size=args.batch_size, 
            num_workers=args.workers,
            bounding_box_json=bbox_json
        )
        
        val_loader = caers_val_data_loader(
            root_dir=args.root_dir, 
            list_file=val_annotation_file_path, 
            image_size=args.image_size, 
            batch_size=args.batch_size, 
            num_workers=args.workers,
            bounding_box_json=bbox_json
        )
        
        test_loader = caers_test_data_loader(
            root_dir=args.root_dir, 
            list_file=test_annotation_file_path, 
            image_size=args.image_size, 
            batch_size=args.batch_size, 
            num_workers=args.workers,
            bounding_box_json=bbox_json
        )
        
        # Weighted Sampler logic for CAER-S
        # Re-implement using the file we just loaded if needed, or rely on Dataloader's internal logic?
        # Standard weighted sampler usually wraps the dataset.
        # But caers_train_data_loader returns a DataLoader. 
        # If we want WeightedRandomSampler, we need to construct it before DataLoader.
        # However, caers_train_data_loader encapsulates DataLoader creation.
        # Let's modify the loader creation to support sampler if needed, 
        # OR just handle it here if we refactor caers_train_data_loader to return dataset.
        # BUT, caers_train_data_loader returns DataLoader directly.
        
        # Quick fix: If we really need weighted sampler, we should modify caers_train_data_loader 
        # or just instantiate dataset here and create loader manually like standard RAER flow.
        # Let's check how standard RAER does it. It creates dataset then loader.
        # For consistency and power, let's instantiate dataset manually here for CAER-S too 
        # to support weighted sampler fully.
        
        print("=> Re-building CAER-S Train Loader for Weighted Sampler support...")
        train_data = CAERSDataset(
            root_dir=args.root_dir, 
            list_file=train_annotation_file_path, 
            mode='train', 
            image_size=args.image_size, 
            bounding_box_json=bbox_json
        )
        
        sampler = None
        shuffle = True
        if args.use_weighted_sampler:
            print("=> Using WeightedRandomSampler for CAER-S.")
            # Calculate class counts from dataset.samples
            # samples is list of (path, label_idx, rel_path)
            targets = [s[1] for s in train_data.samples]
            class_counts = torch.tensor([targets.count(i) for i in range(num_classes)])
            
            # Avoid division by zero
            class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
            
            class_weights = 1. / class_counts.float()
            sample_weights = [class_weights[t] for t in targets]
            
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False
            
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )

        print(f"Total number of training images: {len(train_data)}")
        return train_loader, val_loader, test_loader


    else:
        print(f"Loading train data (Standard) for {args.dataset}...")
        train_data = train_data_loader(
            root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
            bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body,
            num_classes=num_classes
        )
        
        print(f"Loading validation data (Standard) for {args.dataset}...")
        val_data = test_data_loader(
            root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body,
            num_classes=num_classes
        )

        print(f"Loading test data (Standard) for {args.dataset}...")
        test_data = test_data_loader(
            root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
            duration=args.duration, image_size=args.image_size,
            bounding_box_face=args.bounding_box_face,bounding_box_body=args.bounding_box_body,
            crop_body=args.crop_body,
            num_classes=num_classes
        )

    print(f"Total number of training images: {len(train_data)}")
    print("Creating DataLoader instances...")
    
    sampler = None
    shuffle = True
    if args.use_weighted_sampler and args.dataset != "CK+" and args.dataset != "CAER-S": # Disable here for CAER-S as it's handled above
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