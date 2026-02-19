import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from models.Generate_Model import GenerateModel
from models.Text import *
from dataloader.caer_s_dataloader import CAERSDataset
from utils.utils import AverageMeter
from clip import clip

def get_class_info(dataset_name):
    if dataset_name == "CAER" or dataset_name == "CAER-S":
        class_names = class_names_caer
        ensemble_prompts = prompt_ensemble_caer
        return class_names, ensemble_prompts
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported for TTA script.")

def build_model(args, input_text):
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')
    print("Instantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    return model

class TTADataset(CAERSDataset):
    def __init__(self, root_dir, list_file, image_size=224, bounding_box_json=None):
        # Initialize parent without transforms first
        super().__init__(root_dir, list_file, mode='test', image_size=image_size, bounding_box_json=bounding_box_json)
        
        # Override transforms for FiveCrop
        self.transform_body = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.FiveCrop(image_size), # Returns tuple of 5 tensors
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])
        
        self.transform_face = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.FiveCrop(image_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])

    def __getitem__(self, index):
        path, label, rel_path = self.samples[index]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = Image.new('RGB', (self.image_size, self.image_size))

        img_body = img
        img_face = self._get_face_crop(img, rel_path)
        
        # Apply FiveCrop transforms
        # Shape: (5, 3, 224, 224)
        t_body = self.transform_body(img_body)
        t_face = self.transform_face(img_face)
        
        # Add temporal dimension (T=1) -> (5, 1, 3, 224, 224)
        t_body = t_body.unsqueeze(1)
        t_face = t_face.unsqueeze(1)
        
        return t_face, t_body, label

def run_tta(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    class_names, input_text = get_class_info(args.dataset)
    model = build_model(args, input_text)
    model.to(device)
    
    if os.path.isfile(args.checkpoint):
        print(f="=> Loading checkpoint '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Checkpoint loaded.")
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    model.eval()

    # Load Data with TTA
    test_dataset = TTADataset(
        root_dir=args.root_dir,
        list_file=args.test_annotation,
        image_size=args.image_size,
        bounding_box_json=args.bounding_box_face
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True
    )
    
    print(f"Starting TTA Evaluation on {len(test_dataset)} images...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images_face, images_body, target) in enumerate(tqdm(test_loader)):
            # Input shape: (B, 5, 1, 3, 224, 224)
            # Reshape to (B*5, 1, 3, 224, 224) to feed into model
            b, n_crops, t, c, h, w = images_face.shape
            
            images_face = images_face.view(-1, t, c, h, w).to(device)
            images_body = images_body.view(-1, t, c, h, w).to(device)
            target = target.to(device)
            
            # Forward pass
            output, _, _, _ = model(images_face, images_body)
            # Output shape: (B*5, Num_Classes)
            
            # Average predictions across crops
            output = output.view(b, n_crops, -1).mean(dim=1) # (B, Num_Classes)
            
            preds = output.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    cm = confusion_matrix(all_targets, all_preds)
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
    uar = np.nanmean(class_acc) * 100
    war = np.sum(all_preds == all_targets) / len(all_targets) * 100
    
    print("\n" + "="*30)
    print(f"TTA Results (FiveCrop):")
    print(f"WAR (Weighted Accuracy): {war:.2f}%")
    print(f"UAR (Unweighted Accuracy): {uar:.2f}%")
    print("="*30)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    with open(os.path.join(os.path.dirname(args.checkpoint), 'tta_results.txt'), 'w') as f:
        f.write(f"TTA Results (FiveCrop):\n")
        f.write(f"WAR: {war:.2f}%\n")
        f.write(f"UAR: {uar:.2f}%\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CAER-S')
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--test-annotation', type=str, required=True)
    parser.add_argument('--bounding-box-face', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--clip-path', type=str, default='ViT-B/16')
    parser.add_argument('--batch-size', type=int, default=32) # Reduce batch size as inputs are 5x larger
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--gpu', type=str, default='0')
    
    # Model specific args needed for GenerateModel init
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default='end')
    parser.add_argument('--class-specific-contexts', type=str, default='True')
    parser.add_argument('--use-moco', action='store_true') # Add if model was trained with MoCo
    
    args = parser.parse_args()
    run_tta(args)