#!/usr/bin/env python3
"""
DroneDeploy Optimized Dataset
============================

High-performance dataset loader that fixes the performance issues of the original:
- Direct image loading (no complex batch caching)
- Simple file structure
- Fast random access
- Memory efficient
- Compatible with standard PyTorch DataLoaders

This replaces the complex batch-caching system with a simple, fast approach.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
import albumentations as A
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DroneDeployOptimizedDataset(Dataset):
    """
    Optimized DroneDeploy dataset with direct image loading.
    
    Simple structure:
    data_root/
    ├── images/         # patch_0001.jpg, patch_0002.jpg, ...
    ├── masks/          # patch_0001.png, patch_0002.png, ...
    ├── splits/
    │   ├── train.txt   # List of patch IDs for training
    │   ├── val.txt     # List of patch IDs for validation
    │   └── test.txt    # List of patch IDs for testing
    """
    
    # Same class mapping as original
    DRONEDEPLOY_TO_LANDING = {
        81: 2,   # Building → building (avoid)
        91: 0,   # Road → ground (safe primary)
        99: 4,   # Car → car (dynamic obstacle)
        105: 5,  # Background/Clutter → clutter (caution)
        132: 1,  # Trees → vegetation (safe secondary)
        155: 3,  # Pool/Water → water (critical hazard)
        0: 5,    # Unknown → clutter
        255: 5,  # Background → clutter
    }
    
    LANDING_CLASSES = {
        0: "ground",       # Safe flat landing
        1: "vegetation",   # Acceptable emergency landing
        2: "building",     # Hard obstacles to avoid
        3: "water",        # Critical hazard
        4: "car",          # Dynamic obstacles
        5: "clutter"       # Mixed debris/objects
    }
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        patch_size: int = 512,
        augmentation: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.augmentation = augmentation
        
        # Paths
        self.images_dir = self.data_root / 'images'
        self.masks_dir = self.data_root / 'masks'
        self.splits_dir = self.data_root / 'splits'
        
        print(f"🚁 Loading DroneDeploy Optimized dataset for {split}...")
        
        # Create fast lookup table for class mapping
        self.mapping_lut = np.full(256, 5, dtype=np.uint8)
        for dd_class, landing_class in self.DRONEDEPLOY_TO_LANDING.items():
            if dd_class < 256:
                self.mapping_lut[dd_class] = landing_class
        
        # Load split file
        self.patch_ids = self._load_split()
        
        # Setup transforms
        self.transform = self._setup_augmentation()
        
        print(f"   Split {split}: {len(self.patch_ids)} patches")
        print(f"   Images dir: {self.images_dir}")
        print(f"   Masks dir: {self.masks_dir}")
    
    def _load_split(self) -> List[str]:
        """Load patch IDs for this split."""
        split_file = self.splits_dir / f"{self.split}.txt"
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                patch_ids = [line.strip() for line in f if line.strip()]
            print(f"   Loaded {len(patch_ids)} patch IDs from {split_file}")
            return patch_ids
        else:
            print(f"   ⚠️  Split file not found: {split_file}")
            print(f"   📁 Looking for image files directly...")
            
            # Fallback: scan images directory
            image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
            patch_ids = [f.stem for f in image_files]
            
            # Create deterministic splits
            patch_ids.sort()
            total = len(patch_ids)
            
            if self.split == 'train':
                patch_ids = patch_ids[:int(0.6 * total)]
            elif self.split == 'val':
                patch_ids = patch_ids[int(0.6 * total):int(0.8 * total)]
            else:  # test
                patch_ids = patch_ids[int(0.8 * total):]
            
            print(f"   Created {self.split} split with {len(patch_ids)} patches")
            return patch_ids
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup augmentation pipeline."""
        transforms = [
            A.Resize(self.patch_size, self.patch_size, always_apply=True),
        ]
        
        if self.augmentation and self.split == 'train':
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ])
        
        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'}
        )
    
    def __len__(self) -> int:
        return len(self.patch_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a patch by index - simple and fast!"""
        patch_id = self.patch_ids[idx]
        
        # Load image and mask directly
        image_path = self.images_dir / f"{patch_id}.jpg"
        mask_path = self.masks_dir / f"{patch_id}.png"
        
        # Try alternative extensions if not found
        if not image_path.exists():
            image_path = self.images_dir / f"{patch_id}.png"
        if not mask_path.exists():
            mask_path = self.masks_dir / f"{patch_id}.jpg"
        
        # Load files
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Convert and map classes
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.mapping_lut[np.clip(mask, 0, 255)]
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'patch_id': patch_id
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights by sampling the dataset."""
        print("📊 Computing class weights...")
        
        class_counts = np.zeros(6)
        sample_size = min(200, len(self))
        
        for i in tqdm(range(0, sample_size), desc="Sampling"):
            try:
                sample = self[i]
                mask = sample['mask'].numpy()
                
                for class_id in range(6):
                    class_counts[class_id] += np.sum(mask == class_id)
            except:
                continue
        
        # Compute inverse frequency weights
        total_pixels = np.sum(class_counts)
        class_weights = torch.ones(6)
        
        for class_id in range(6):
            count = max(class_counts[class_id], 1)  # Avoid division by zero
            class_weights[class_id] = total_pixels / (6 * count)
        
        # Apply safety multipliers
        class_weights[3] *= 1.5  # Water
        class_weights[4] *= 2.0  # Car
        
        # Normalize and clip
        class_weights = torch.clamp(class_weights, min=0.5, max=5.0)
        
        return class_weights


def create_optimized_dronedeploy_datasets(
    data_root: str,
    patch_size: int = 512,
    **kwargs
) -> Dict[str, DroneDeployOptimizedDataset]:
    """
    Create optimized DroneDeploy datasets.
    
    Args:
        data_root: Root directory with images/, masks/, splits/ structure
        patch_size: Target patch size
        **kwargs: Additional dataset parameters
        
    Returns:
        Dictionary with train/val/test datasets
    """
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = DroneDeployOptimizedDataset(
                data_root=data_root,
                split=split,
                patch_size=patch_size,
                **kwargs
            )
            datasets[split] = dataset
        except Exception as e:
            print(f"❌ Failed to create {split} dataset: {e}")
            datasets[split] = None
    
    return datasets


if __name__ == "__main__":
    # Test the optimized dataset
    print("🚁 Testing DroneDeploy Optimized Dataset...")
    
    # Example usage
    try:
        datasets = create_optimized_dronedeploy_datasets(
            data_root="./datasets/dronedeploy_optimized",
            patch_size=512,
            augmentation=True
        )
        
        print(f"\nDataset creation successful!")
        for split, dataset in datasets.items():
            if dataset is not None:
                print(f"   {split}: {len(dataset)} patches")
        
        # Test sample loading
        if datasets['train'] is not None and len(datasets['train']) > 0:
            sample = datasets['train'][0]
            print(f"\n📋 Sample test:")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Mask shape: {sample['mask'].shape}")
            print(f"   Unique classes: {torch.unique(sample['mask'])}")
            print(f"   Patch ID: {sample['patch_id']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 