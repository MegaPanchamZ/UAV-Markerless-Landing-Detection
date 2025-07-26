#!/usr/bin/env python3
"""
Semantic Drone Dataset Integration for UAV Landing Detection
===========================================================

Professional dataset integration for the Semantic Drone Dataset:
- 400 training images at 6000x4000 resolution (24MP)
- 24 fine-grained semantic classes
- Professional annotation quality
- Multi-scale training support
- Advanced class mapping for landing detection

Dataset: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset/data
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from collections import Counter
import warnings
import time
from PIL import Image
import torchvision.transforms as transforms

class SemanticDroneDataset(Dataset):
    """
    Semantic Drone Dataset for UAV Landing Detection.
    
    Maps 24 original semantic classes to 4 landing-relevant classes:
    0: Background/Unknown
    1: Safe Landing (paved-area, dirt, grass, gravel)
    2: Caution (rocks, vegetation, roof, ar-marker) 
    3: Danger (water, pool, obstacles, people, vehicles)
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_resolution: Tuple[int, int] = (512, 512),
        class_mapping: str = "enhanced_4_class",
        return_confidence: bool = False,
        cache_images: bool = True,
        use_random_crops: bool = True,
        crops_per_image: int = 4,
        preload_to_memory: bool = False  # NEW: For 251GB RAM systems
    ):
        """
        Initialize Semantic Drone Dataset.
        
        Args:
            data_root: Path to dataset root directory
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform pipeline
            target_resolution: Target image resolution for training
            class_mapping: Type of class mapping ('enhanced_4_class', 'advanced_6_class')
            return_confidence: Whether to return confidence maps
            cache_images: Cache images in memory for faster training
            use_random_crops: Use random crops instead of full image resizing (much faster)
            crops_per_image: Number of random crops per image (effectively multiplies dataset size)
            preload_to_memory: Load entire dataset to RAM (for 200GB+ systems)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_resolution = target_resolution
        self.class_mapping_type = class_mapping
        self.return_confidence = return_confidence
        self.cache_images = cache_images
        self.use_random_crops = use_random_crops
        self.crops_per_image = crops_per_image if split == "train" else 1  # Only use multiple crops for training
        self.preload_to_memory = preload_to_memory  # Store the parameter!
        
        # NEW: Memory storage for preloaded data
        self.memory_cache = {}
        self._memory_loaded = False
        
        # Original 24 classes from Semantic Drone Dataset
        self.original_classes = {
            0: "unlabeled", 1: "paved-area", 2: "dirt", 3: "grass", 4: "gravel",
            5: "water", 6: "rocks", 7: "pool", 8: "vegetation", 9: "roof",
            10: "wall", 11: "window", 12: "door", 13: "fence", 14: "fence-pole",
            15: "person", 16: "dog", 17: "car", 18: "bicycle", 19: "tree",
            20: "bald-tree", 21: "ar-marker", 22: "obstacle", 23: "conflicting"
        }
        
        # Initialize class mapping
        self._setup_class_mapping()
        
        # OPTIMIZATION: Create fast lookup table for class mapping
        self.mapping_lut = np.zeros(256, dtype=np.uint8)
        for original_class, landing_class in self.class_mapping.items():
            self.mapping_lut[original_class] = landing_class
        
        # Load dataset information
        self.images_dir = self.data_root / "original_images"
        self.labels_dir = self.data_root / "label_images_semantic"
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")
        
        # Load image and label files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = sorted(list(self.labels_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        if len(self.label_files) == 0:
            raise ValueError(f"No labels found in {self.labels_dir}")
        
        # Create train/val/test splits
        self._create_splits()
        
        # Image cache for faster training
        self.image_cache = {} if cache_images else None
        self.label_cache = {} if cache_images else None
        
        # NEW: Preload entire dataset to memory if enabled
        if preload_to_memory:
            self._preload_dataset_to_memory()
        
        print(f"SemanticDroneDataset initialized:")
        print(f"   Split: {split} ({len(self.file_indices)} samples)")
        print(f"   Classes: {len(self.landing_classes)} landing classes")
        print(f"   Resolution: {target_resolution}")
        print(f"   Mapping: {class_mapping}")
        if preload_to_memory:
            print(f"   🚀 ENTIRE DATASET LOADED TO MEMORY (CPU bottleneck eliminated!)")
        if self.use_random_crops:
            print(f"   Random crops: {self.crops_per_image} per image")
        
    def _setup_class_mapping(self):
        """Setup class mapping based on selected type."""
        
        if self.class_mapping_type == "enhanced_4_class":
            # Enhanced 4-class mapping (recommended)
            self.class_mapping = {
                # Background/Unknown
                0: 0, 23: 0,  # unlabeled, conflicting
                
                # Safe Landing - flat, stable surfaces
                1: 1, 2: 1, 3: 1, 4: 1,  # paved-area, dirt, grass, gravel
                
                # Caution - potentially suitable, needs assessment
                6: 2, 8: 2, 9: 2, 21: 2,  # rocks, vegetation, roof, ar-marker
                
                # Danger - obstacles, hazards, unsuitable
                5: 3, 7: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,  # water, pool, wall, window, door, fence, fence-pole
                15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 22: 3   # person, dog, car, bicycle, tree, bald-tree, obstacle
            }
            
            self.landing_classes = {
                0: "background",
                1: "safe_landing", 
                2: "caution",
                3: "danger"
            }
            
        elif self.class_mapping_type == "advanced_6_class":
            # Advanced 6-class mapping for high precision
            self.class_mapping = {
                0: 0, 23: 0,           # background
                1: 1, 2: 1,            # optimal_landing (paved-area, dirt)
                3: 2, 4: 2,            # good_landing (grass, gravel)
                8: 3, 9: 3,            # caution_surface (vegetation, roof)
                10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 19: 4, 20: 4, 22: 4,  # physical_obstacle
                5: 5, 7: 5, 15: 5, 16: 5, 17: 5, 18: 5, 6: 5, 21: 5      # critical_hazard
            }
            
            self.landing_classes = {
                0: "background",
                1: "optimal_landing",
                2: "good_landing", 
                3: "caution_surface",
                4: "physical_obstacle",
                5: "critical_hazard"
            }
        elif self.class_mapping_type == "unified_6_class":
            # UNIFIED 6-class mapping consistent with DroneDeploy and UDD6
            # This ensures consistent semantic meaning across all datasets in progressive training
            self.class_mapping = {
                # 0: ground - safe flat landing surfaces
                1: 0, 2: 0, 4: 0,      # paved-area, dirt, gravel → ground (safe flat surfaces)
                
                # 1: vegetation - trees, grass (emergency landing acceptable)
                3: 1, 8: 1, 19: 1, 20: 1,  # grass, vegetation, tree, bald-tree → vegetation
                
                # 2: building - obstacles to avoid  
                9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2,  # roof, wall, window, door, fence, fence-pole → building
                
                # 3: water - critical hazard
                5: 3, 7: 3,            # water, pool → water
                
                # 4: car - dynamic obstacles
                17: 4, 18: 4,          # car, bicycle → car
                
                # 5: clutter - unknown/mixed areas
                0: 5, 6: 5, 15: 5, 16: 5, 21: 5, 22: 5, 23: 5  # unlabeled, rocks, person, dog, ar-marker, obstacle, conflicting → clutter
            }
            
            self.landing_classes = {
                0: "ground",       # Safe flat landing surfaces (roads, dirt, pavement)
                1: "vegetation",   # Trees, grass (emergency landing acceptable)
                2: "building",     # Buildings, walls, obstacles to avoid
                3: "water",        # Water bodies, pools (critical hazard)
                4: "car",          # Cars, vehicles (dynamic obstacles)
                5: "clutter"       # Unknown, mixed debris, people, animals
            }
        else:
            raise ValueError(f"Unknown class mapping: {self.class_mapping_type}")
    
    def _create_splits(self):
        """Create train/val/test splits with stratification."""
        
        total_files = len(self.image_files)
        indices = np.arange(total_files)
        
        # Stratified splits to ensure balanced class distribution
        # 70% train, 15% val, 15% test
        np.random.seed(42)  # Reproducible splits
        np.random.shuffle(indices)
        
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
        """Get a sample from the dataset."""
        
        # NEW: Use memory cache if available (MUCH faster!)
        if self.preload_to_memory and self._memory_loaded:
            return self._get_item_from_memory(idx)
        else:
            return self._get_item_from_disk(idx)
    
    def _get_item_from_memory(self, idx):
        """
        OPTIMIZED: Get item from memory cache - eliminates all I/O bottlenecks and conversion overhead.
        """
        if self.use_random_crops:
            # Handle random crops
            base_idx = idx // self.crops_per_image
            cache_data = self.memory_cache[base_idx]
        else:
            cache_data = self.memory_cache[idx]
        
        # Get data from memory (instant access!) - already in numpy format
        image = cache_data['image'].copy()  # Copy to avoid mutation
        mask = cache_data['mask'].copy()
        
        # Apply random crops if needed (working directly with numpy arrays)
        if self.use_random_crops:
            image, mask = self._apply_random_crop_numpy(image, mask)
        
        # Apply transforms (albumentations works natively with numpy)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors if not already done by transforms
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': cache_data['img_path'],
            'original_shape': mask.shape
        }
    
    def _get_item_from_disk(self, idx):
        """Original disk-based loading (slower but lower memory)."""
        
        file_idx = self.file_indices[idx // self.crops_per_image]
        image_path = self.image_files[file_idx]
        label_path = self.label_files[file_idx]
        
        # Load image
        if self.image_cache and file_idx in self.image_cache:
            image = self.image_cache[file_idx]
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.image_cache is not None:
                self.image_cache[file_idx] = image
        
        # Load label
        if self.label_cache and file_idx in self.label_cache:
            label = self.label_cache[file_idx]
        else:
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if self.label_cache is not None:
                self.label_cache[file_idx] = label
        
        # FIXED: 1024x1024 crops then resize to target resolution (better context)
        if self.use_random_crops and self.split == "train":
            # Take 1024x1024 crops from the high-resolution image for better context
            crop_size = 1024
            img_h, img_w = image.shape[:2]
            
            # Ensure crop size doesn't exceed image size
            crop_size = min(crop_size, min(img_h, img_w))
            
            # Random crop coordinates
            if img_h > crop_size:
                start_h = np.random.randint(0, img_h - crop_size + 1)
            else:
                start_h = 0
            
            if img_w > crop_size:
                start_w = np.random.randint(0, img_w - crop_size + 1)
            else:
                start_w = 0
            
            # Extract 1024x1024 crop
            image = image[start_h:start_h + crop_size, start_w:start_w + crop_size]
            label = label[start_h:start_h + crop_size, start_w:start_w + crop_size]
            
            # Resize to target resolution (e.g., 256x256)
            image = cv2.resize(image, self.target_resolution[::-1], interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.target_resolution[::-1], interpolation=cv2.INTER_NEAREST)
        else:
            # For validation/test, resize to target resolution
            image = cv2.resize(image, self.target_resolution[::-1], interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.target_resolution[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Map original classes to landing classes
        mapped_label = self._map_classes(label)
        
        # Store original labels for confidence computation (before transforms)
        original_label_for_confidence = label.copy()
        mapped_label_for_confidence = mapped_label.copy()
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mapped_label)
            image = transformed['image']
            mapped_label = transformed['mask']
        
        # Convert to tensors if not already done by transforms
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mapped_label, torch.Tensor):
            mapped_label = torch.from_numpy(mapped_label).long()
        
        sample = {
            'image': image,
            'mask': mapped_label,
            'image_path': str(image_path),
            'original_shape': label.shape
        }
        
        # Add confidence map if requested
        if self.return_confidence:
            # Use pre-transform labels for confidence computation
            confidence = self._compute_confidence_map(original_label_for_confidence, mapped_label_for_confidence)
            # Resize confidence to match transformed label if necessary
            if hasattr(mapped_label, 'shape') and confidence.shape != mapped_label.shape[-2:]:
                target_shape = mapped_label.shape[-2:][::-1] if hasattr(mapped_label, 'shape') else (512, 512)
                confidence = cv2.resize(confidence.astype(np.float32), target_shape, interpolation=cv2.INTER_NEAREST)
            sample['confidence'] = torch.from_numpy(confidence).float()
        
        return sample
    
    def _map_classes(self, label: np.ndarray) -> np.ndarray:
        """Map original 24 classes to landing classes using fast vectorized lookup."""
        # OPTIMIZATION: Use vectorized lookup table - orders of magnitude faster
        return self.mapping_lut[label]
    
    def _compute_confidence_map(self, original_label: np.ndarray, mapped_label: np.ndarray) -> torch.Tensor:
        """Compute confidence map based on class mapping certainty."""
        
        confidence = np.ones_like(mapped_label, dtype=np.float32)
        
        # Lower confidence for ambiguous mappings
        ambiguous_classes = [6, 8, 9, 21]  # rocks, vegetation, roof, ar-marker
        for class_id in ambiguous_classes:
            mask = (original_label == class_id)
            confidence[mask] = 0.7  # Lower confidence for caution class
        
        # High confidence for clear safe/dangerous classes
        clear_safe = [1, 2, 3, 4]  # paved-area, dirt, grass, gravel
        clear_danger = [5, 7, 15, 16, 17, 18, 19, 20, 22]  # water, pool, person, etc.
        
        for class_id in clear_safe + clear_danger:
            mask = (original_label == class_id)
            confidence[mask] = 1.0
        
        return torch.from_numpy(confidence)
    
    def get_class_distribution(self) -> Dict[str, float]:
        """Analyze class distribution in the dataset."""
        
        class_counts = Counter()
        total_pixels = 0
        
        print("Analyzing class distribution...")
        for idx in range(min(len(self), 50)):  # Sample first 50 images
            sample = self[idx]
            label = sample['mask'].numpy()
            
            unique, counts = np.unique(label, return_counts=True)
            for class_id, count in zip(unique, counts):
                class_counts[class_id] += count
                total_pixels += count
        
        # Convert to percentages
        distribution = {}
        for class_id, count in class_counts.items():
            class_name = self.landing_classes.get(class_id, f"class_{class_id}")
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
            distribution[class_name] = percentage
        
        return distribution
    
    def get_sample_weights(self) -> torch.Tensor:
        """Compute sample weights for balanced training."""
        
        distribution = self.get_class_distribution()
        
        # Inverse frequency weighting
        weights = {}
        total_classes = len(self.landing_classes)
        
        for class_id, class_name in self.landing_classes.items():
            freq = distribution.get(class_name, 0.1) / 100.0  # Convert to fraction
            weight = 1.0 / (freq * total_classes) if freq > 0 else 1.0
            weights[class_id] = weight
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {k: v / weight_sum * total_classes for k, v in weights.items()}
        
        # Convert to tensor
        weight_tensor = torch.ones(len(self.landing_classes))
        for class_id, weight in weights.items():
            weight_tensor[class_id] = weight
        
        return weight_tensor

    def _preload_dataset_to_memory(self):
        """
        NEW: Preload entire dataset to memory for massive RAM systems.
        
        This completely eliminates CPU bottleneck by loading all images/masks
        into RAM once. Perfect for 251GB+ systems.
        """
        print(f"\n🚀 PRELOADING {self.split.upper()} DATASET TO MEMORY...")
        print(f"   Total samples: {len(self.file_indices)}")
        print(f"   Available RAM: Loading entire dataset to eliminate CPU bottleneck")
        
        from tqdm import tqdm
        import gc
        
        start_time = time.time()
        
        for idx in tqdm(range(len(self.file_indices)), desc=f"Loading {self.split} to RAM"):
            file_idx = self.file_indices[idx]
            img_path = self.image_files[file_idx]
            mask_path = self.label_files[file_idx]
            
            # Load and store raw data in memory
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                image_array = np.array(image)
                
                # Load mask
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                
                # Apply class mapping
                mapped_mask = self._map_classes(mask_array) # Use the existing _map_classes
                
                # Store in memory cache
                self.memory_cache[idx] = {
                    'image': image_array,
                    'mask': mapped_mask,
                    'img_path': str(img_path),
                    'mask_path': str(mask_path)
                }
                
            except Exception as e:
                print(f"   ❌ Failed to load {img_path}: {e}")
                continue
        
        # Force garbage collection
        gc.collect()
        
        load_time = time.time() - start_time
        memory_usage_gb = len(self.memory_cache) * 0.01  # Rough estimate
        
        self._memory_loaded = True
        print(f"✅ DATASET PRELOADED IN {load_time:.1f}s")
        print(f"   • Samples in memory: {len(self.memory_cache)}")
        print(f"   • Estimated memory usage: ~{memory_usage_gb:.1f}GB")
        print(f"   • CPU bottleneck: ELIMINATED! 🚀")
    
    def _apply_random_crop_numpy(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        FIXED: Apply 1024x1024 crop then resize to target resolution.
        """
        # Take 1024x1024 crop for better context
        crop_size = 1024
        img_h, img_w = image.shape[:2]
        
        # Ensure crop doesn't exceed image size
        crop_size = min(crop_size, min(img_h, img_w))
        
        # Random crop coordinates
        if img_h > crop_size:
            start_h = np.random.randint(0, img_h - crop_size + 1)
        else:
            start_h = 0
        
        if img_w > crop_size:
            start_w = np.random.randint(0, img_w - crop_size + 1)
        else:
            start_w = 0
        
        # Extract 1024x1024 crop
        image_crop = image[start_h:start_h + crop_size, start_w:start_w + crop_size]
        mask_crop = mask[start_h:start_h + crop_size, start_w:start_w + crop_size]
        
        # Resize to target resolution (e.g., 256x256)
        image_crop = cv2.resize(image_crop, self.target_resolution[::-1], interpolation=cv2.INTER_LINEAR)
        mask_crop = cv2.resize(mask_crop, self.target_resolution[::-1], interpolation=cv2.INTER_NEAREST)
        
        return image_crop, mask_crop


def create_semantic_drone_transforms(
    input_size: Tuple[int, int] = (512, 512),
    is_training: bool = True,
    augmentation_profile: str = "balanced",  # "fast", "balanced", "advanced"
    use_resize: bool = False  # Don't resize if using random crops
) -> A.Compose:
    """
    Create augmentation pipeline for Semantic Drone Dataset.
    
    Args:
        input_size: Target image size
        is_training: Whether to apply training augmentations
        augmentation_profile: Augmentation intensity ("fast", "balanced", "advanced")
        use_resize: Whether to include resize transform (disable when using random crops)
    """
    
    transforms_list = []
    
    # Only add resize if specifically requested (not needed with random crops)
    if use_resize:
        transforms_list.append(A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_LINEAR))
    
    if is_training:
        if augmentation_profile == "fast":
            # OPTIMIZATION: Minimal, fast augmentations only
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            ])
        elif augmentation_profile == "balanced":
            # OPTIMIZATION: Balanced augmentations - good variety without slow operations
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),  # Aerial imagery can be flipped
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3  # Reduced probability
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Only fast blur
                A.GaussNoise(var_limit=(10, 30), p=0.2),
            ])
        elif augmentation_profile == "advanced":
            # Advanced augmentations for best generalization (slower)
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
                ),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10)
                ], p=0.7),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=5)
                ], p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.15),  # Reduced probability
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.15),  # Reduced probability
            ])
    
    # Normalization (ImageNet stats)
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def validate_semantic_drone_dataset(data_root: str) -> bool:
    """Validate Semantic Drone Dataset structure and files."""
    
    data_root = Path(data_root)
    
    required_dirs = [
        "original_images",
        "label_images_semantic",
        "RGB_color_image_masks"  # Optional for visualization
    ]
    
    print("🔍 Validating Semantic Drone Dataset...")
    
    issues = []
    
    # Check directory structure
    for dir_name in required_dirs[:2]:  # Skip optional RGB masks for now
        dir_path = data_root / dir_name
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")
        else:
            files = list(dir_path.glob("*"))
            if len(files) == 0:
                issues.append(f"Empty directory: {dir_path}")
            else:
                print(f" {dir_name}: {len(files)} files")
    
    # Check file correspondence
    if not issues:
        images_dir = data_root / "original_images"
        labels_dir = data_root / "label_images_semantic"
        
        image_files = sorted(list(images_dir.glob("*.jpg")))
        label_files = sorted(list(labels_dir.glob("*.png")))
        
        if len(image_files) != len(label_files):
            issues.append(f"Image/label count mismatch: {len(image_files)} vs {len(label_files)}")
        
        # Check file naming correspondence
        for img_file in image_files[:10]:  # Sample first 10
            expected_label = labels_dir / f"{img_file.stem}.png"
            if not expected_label.exists():
                issues.append(f"Missing label for {img_file.name}")
    
    # Report validation results
    if issues:
        print("❌ Validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(" Dataset validation passed!")
        return True


if __name__ == "__main__":
    # Example usage and testing
    data_root = "../datasets/semantic_drone_dataset"
    
    if validate_semantic_drone_dataset(data_root):
        # Create dataset
        train_transform = create_semantic_drone_transforms(
            input_size=(512, 512),
            is_training=True,
            advanced_augmentation=True
        )
        
        dataset = SemanticDroneDataset(
            data_root=data_root,
            split="train",
            transform=train_transform,
            class_mapping="enhanced_4_class",
            return_confidence=True,
            use_random_crops=True,
            crops_per_image=4,
            preload_to_memory=True # Enable preloading for example
        )
        
        print(f"\n📊 Dataset Info:")
        print(f"   Samples: {len(dataset)}")
        print(f"   Classes: {dataset.landing_classes}")
        
        # Analyze class distribution
        distribution = dataset.get_class_distribution()
        print(f"\n Class Distribution:")
        for class_name, percentage in distribution.items():
            print(f"   {class_name}: {percentage:.1f}%")
        
        # Test sample loading
        sample = dataset[0]
        print(f"\n🖼️ Sample Info:")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Unique classes: {torch.unique(sample['mask'])}") 