#!/usr/bin/env python3
"""
Urban Drone Dataset (UDD6) Loader with Proper RGB Color Mapping
===============================================================

UDD6 dataset for Stage 3 domain adaptation in progressive training:
- 200+ training images at 3840×2160+ resolution  
- 6 classes using proper RGB color mapping as per UDD6 specification
- High-altitude perspective (60-100m)
- Dense urban environments
- Domain adaptation for altitude/urban robustness
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
import warnings

class UDD6Dataset(Dataset):
    """
    Urban Drone Dataset (UDD6) for Stage 3 domain adaptation.
    Uses proper RGB color mapping as per UDD6 specification.
    Maps UDD6's 6 classes to unified landing classes for progressive training.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_resolution: Tuple[int, int] = (512, 512),
        cache_images: bool = True,
        use_random_crops: bool = True,
        crops_per_image: int = 4
    ):
        """
        Initialize UDD6 Dataset for domain adaptation.
        
        Args:
            data_root: Path to UDD6 dataset root
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            target_resolution: Target image resolution
            cache_images: Cache images in memory
            use_random_crops: Use random crops for data augmentation
            crops_per_image: Number of crops per image
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_resolution = target_resolution
        self.cache_images = cache_images
        self.use_random_crops = use_random_crops
        self.crops_per_image = crops_per_image if split == "train" else 1
        
        # UDD6 RGB color to class mapping (as per UDD6 specification)
        self.RGB_TO_CLASS = {
            (102, 102, 156): 1,  # facade
            (128, 64, 128): 2,   # road
            (107, 142, 35): 3,   # vegetation
            (0, 0, 142): 4,      # vehicle
            (70, 70, 70): 5,     # roof
            (0, 0, 0): 0,        # other/background
        }
        
        # UDD6 class definitions
        self.udd6_classes = {
            0: {"name": "other", "rgb": (0, 0, 0)},
            1: {"name": "facade", "rgb": (102, 102, 156)}, 
            2: {"name": "road", "rgb": (128, 64, 128)},
            3: {"name": "vegetation", "rgb": (107, 142, 35)},
            4: {"name": "vehicle", "rgb": (0, 0, 142)},
            5: {"name": "roof", "rgb": (70, 70, 70)}
        }
        
        # Map UDD6 classes to unified landing classes (6 classes)
        self.class_mapping = {
            0: 5,  # other → other
            1: 2,  # facade → obstacle  
            2: 0,  # road → ground
            3: 1,  # vegetation → vegetation
            4: 4,  # vehicle → vehicle
            5: 2   # roof → obstacle
        }
        
        self.landing_classes = {
            0: "ground",       # Safe flat surfaces
            1: "vegetation",   # Grass, trees (emergency acceptable)
            2: "obstacle",     # Buildings, walls, roofs (avoid)
            3: "water",        # Water bodies (critical hazard) - not in UDD6
            4: "vehicle",      # Cars, moving objects (dynamic avoid)
            5: "other"         # Clutter, unknown areas
        }
        
        # Create fast RGB to class lookup table (3D array for vectorized lookup)
        self._create_rgb_lookup_table()
        
        # OPTIMIZATION: Create fast lookup table for class mapping
        self.mapping_lut = np.zeros(256, dtype=np.uint8)
        for udd6_class, landing_class in self.class_mapping.items():
            self.mapping_lut[udd6_class] = landing_class
        
        # Find dataset files
        self._find_dataset_files()
        self._create_splits()
        
        # Image cache
        self.image_cache = {} if cache_images else None
        self.label_cache = {} if cache_images else None
        
        print(f"🏙️  UDD6Dataset initialized:")
        print(f"   Split: {split} ({len(self.file_indices)} images)")
        print(f"   Effective samples: {len(self)} (with crops)")
        print(f"   Classes: 6 → 6 landing classes")
        print(f"   Resolution: {target_resolution}")
        print(f"   Domain: High-altitude urban")
    
    def _create_rgb_lookup_table(self):
        """Create fast RGB to class ID lookup table."""
        # Create a 3D lookup table for RGB values
        # This is memory intensive but very fast for conversion
        self.rgb_lookup = np.full((256, 256, 256), 0, dtype=np.uint8)  # Default to other/background
        
        for rgb_color, class_id in self.RGB_TO_CLASS.items():
            r, g, b = rgb_color
            self.rgb_lookup[r, g, b] = class_id
        
        print(f"   Created RGB lookup table for {len(self.RGB_TO_CLASS)} color mappings")
    
    def _rgb_to_class_vectorized(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image to class IDs using vectorized lookup."""
        h, w = rgb_image.shape[:2]
        
        # Clip RGB values to valid range
        rgb_clipped = np.clip(rgb_image, 0, 255)
        
        # Use advanced indexing for vectorized lookup
        class_image = self.rgb_lookup[rgb_clipped[:, :, 0], 
                                     rgb_clipped[:, :, 1], 
                                     rgb_clipped[:, :, 2]]
        
        return class_image
    
    def _find_dataset_files(self):
        """Find UDD6 dataset files in various possible structures."""
        possible_structures = [
            # Structure 1: train/val/test dirs with src/gt subdirs
            {
                "images": ["train/src", "val/src", "test/src"],
                "labels": ["train/gt", "val/gt", "test/gt"]
            },
            # Structure 2: separate train/val with images/labels
            {
                "images": ["images", "src", "train_images"],
                "labels": ["labels", "gt", "train_labels", "annotations"]
            },
            # Structure 3: flat structure
            {
                "images": [".", "images"],
                "labels": [".", "labels", "gt"]
            }
        ]
        
        self.image_files = []
        self.label_files = []
        
        for structure in possible_structures:
            for img_dir in structure["images"]:
                img_path = self.data_root / img_dir
                if img_path.exists():
                    imgs = list(img_path.glob("*.jpg")) + list(img_path.glob("*.JPG")) + list(img_path.glob("*.png")) + list(img_path.glob("*.PNG"))
                    if imgs:
                        self.image_files = sorted(imgs)
                        print(f"   Found images in: {img_path}")
                        break
            
            if self.image_files:
                for lbl_dir in structure["labels"]:
                    lbl_path = self.data_root / lbl_dir
                    if lbl_path.exists():
                        lbls = list(lbl_path.glob("*.png")) + list(lbl_path.glob("*.PNG")) + list(lbl_path.glob("*.jpg")) + list(lbl_path.glob("*.JPG"))
                        if lbls:
                            self.label_files = sorted(lbls)
                            print(f"   Found labels in: {lbl_path}")
                            break
                break
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.data_root}")
        if not self.label_files:
            raise ValueError(f"No labels found in {self.data_root}")
        
        # Verify correspondence
        if len(self.image_files) != len(self.label_files):
            print(f"   ⚠️  Image/label count mismatch: {len(self.image_files)} vs {len(self.label_files)}")
            # Try to match by filename
            self._match_files_by_name()
    
    def _match_files_by_name(self):
        """Match image and label files by filename."""
        matched_pairs = []
        
        for img_file in self.image_files:
            # Try different naming patterns
            possible_label_names = [
                img_file.stem + ".png",
                img_file.stem + "_gt.png",
                img_file.stem + "_label.png",
                img_file.name  # Same name different extension
            ]
            
            for label_file in self.label_files:
                if label_file.name in possible_label_names:
                    matched_pairs.append((img_file, label_file))
                    break
        
        if len(matched_pairs) < len(self.image_files) * 0.8:
            raise ValueError(f"Could not match enough image/label pairs: {len(matched_pairs)} out of {len(self.image_files)}")
        
        self.image_files, self.label_files = zip(*matched_pairs)
        self.image_files = list(self.image_files)
        self.label_files = list(self.label_files)
        
        print(f"   Matched {len(matched_pairs)} image/label pairs")
    
    def _create_splits(self):
        """Create train/val/test splits."""
        total_files = len(self.image_files)
        indices = np.arange(total_files)
        
        # Reproducible splits
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # 70% train, 15% val, 15% test
        train_end = int(0.7 * total_files)
        val_end = int(0.85 * total_files)
        
        if self.split == "train":
            self.file_indices = indices[:train_end]
        elif self.split == "val":
            self.file_indices = indices[train_end:val_end]
        elif self.split == "test":
            self.file_indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self):
        return len(self.file_indices) * self.crops_per_image
    
    def __getitem__(self, idx):
        # Determine which image and crop
        image_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image
        
        file_idx = self.file_indices[image_idx]
        image_path = self.image_files[file_idx]
        label_path = self.label_files[file_idx]
        
        # Load image
        if self.image_cache is not None and file_idx in self.image_cache:
            image = self.image_cache[file_idx]
        else:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.image_cache is not None:
                self.image_cache[file_idx] = image
        
        # Load label
        if self.label_cache is not None and file_idx in self.label_cache:
            label_rgb = self.label_cache[file_idx]
        else:
            label_rgb = cv2.imread(str(label_path))
            if label_rgb is None:
                raise ValueError(f"Could not load label: {label_path}")
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            if self.label_cache is not None:
                self.label_cache[file_idx] = label_rgb
        
        # Random cropping for data augmentation
        if self.use_random_crops and self.split == "train":
            image, label_rgb = self._random_crop(image, label_rgb, crop_idx)
        else:
            # Resize for validation/test
            image = cv2.resize(image, self.target_resolution, interpolation=cv2.INTER_LINEAR)
            label_rgb = cv2.resize(label_rgb, self.target_resolution, interpolation=cv2.INTER_NEAREST)
        
        # Convert RGB labels to UDD6 class IDs using vectorized lookup
        udd6_label = self._rgb_to_class_vectorized(label_rgb)
        
        # Convert UDD6 classes to landing classes
        landing_label = self._map_to_landing_classes(udd6_label)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=landing_label)
            image = transformed['image']
            landing_label = transformed['mask']
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(landing_label, torch.Tensor):
            landing_label = torch.from_numpy(landing_label).long()
        
        return {
            'image': image,
            'mask': landing_label,
            'image_path': str(image_path),
            'dataset_source': 'udd6'
        }
    
    def _random_crop(self, image: np.ndarray, label_rgb: np.ndarray, crop_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random crop for data augmentation."""
        crop_h, crop_w = self.target_resolution
        img_h, img_w = image.shape[:2]
        
        # Ensure crop doesn't exceed image size
        crop_h = min(crop_h, img_h)
        crop_w = min(crop_w, img_w)
        
        # Deterministic crop for validation
        if self.split != "train":
            np.random.seed(crop_idx)
        
        # Random crop coordinates
        if img_h > crop_h:
            start_h = np.random.randint(0, img_h - crop_h + 1)
        else:
            start_h = 0
        
        if img_w > crop_w:
            start_w = np.random.randint(0, img_w - crop_w + 1)
        else:
            start_w = 0
        
        # Extract crop
        image_crop = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
        label_crop = label_rgb[start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        return image_crop, label_crop
    
    def _map_to_landing_classes(self, udd6_label: np.ndarray) -> np.ndarray:
        """Map UDD6 classes to unified landing classes using fast vectorized lookup."""
        # OPTIMIZATION: Use vectorized lookup table - orders of magnitude faster
        return self.mapping_lut[udd6_label]
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        class_counts = Counter()
        
        # Sample subset for analysis
        sample_size = min(len(self.file_indices), 20)
        
        for i in range(sample_size):
            file_idx = self.file_indices[i]
            label_path = self.label_files[file_idx]
            
            # Load and process label
            label_rgb = cv2.imread(str(label_path))
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            
            # Convert RGB to UDD6 classes using vectorized lookup
            udd6_label = self._rgb_to_class_vectorized(label_rgb)
            
            # Convert to landing classes
            landing_label = self._map_to_landing_classes(udd6_label)
            
            unique, counts = np.unique(landing_label, return_counts=True)
            for class_id, count in zip(unique, counts):
                class_counts[class_id] += count
        
        # Compute inverse frequency weights
        total_pixels = sum(class_counts.values())
        class_weights = torch.ones(6)
        
        for class_id in range(6):
            count = class_counts.get(class_id, 1)
            class_weights[class_id] = total_pixels / (6 * count)
        
        # Normalize and clip
        class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
        
        return class_weights


def create_udd6_transforms(
    input_size: Tuple[int, int] = (512, 512),
    is_training: bool = True
) -> A.Compose:
    """Create transforms optimized for high-altitude urban drone imagery."""
    
    transforms = []
    
    if is_training:
        # Urban-specific augmentations
        transforms.extend([
            A.RandomRotate90(p=0.8),  # All rotations valid for aerial
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.8, brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(p=0.6, hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.1),  # Simulate shadow/fog effects
            A.GaussianBlur(p=0.2, blur_limit=(3, 5)),  # Atmospheric effects
            A.GaussNoise(p=0.3, var_limit=(10, 30)),
        ])
    
    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms)


if __name__ == "__main__":
    # Test UDD6 dataset loading
    print("🏙️  Testing UDD6 Dataset...")
    
    try:
        data_root = "../datasets/udd6"  # Adjust path as needed
        
        transforms = create_udd6_transforms(is_training=True)
        
        dataset = UDD6Dataset(
            data_root=data_root,
            split="train",
            transform=transforms,
            use_random_crops=True,
            crops_per_image=4
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Dataset size: {len(dataset)} samples")
        
        # Test sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample image shape: {sample['image'].shape}")
            print(f"   Sample mask shape: {sample['mask'].shape}")
            print(f"   Unique classes: {torch.unique(sample['mask'])}")
            print(f"   Dataset source: {sample['dataset_source']}")
        
        # Test class weights
        class_weights = dataset.get_class_weights()
        print(f"   Class weights: {class_weights}")
        
    except Exception as e:
        print(f"❌ UDD6 Dataset test failed: {e}") 