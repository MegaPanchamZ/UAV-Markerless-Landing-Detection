#!/usr/bin/env python3
"""
DroneDeploy 1024 Dataset with Caching
====================================

High-performance dataset loader for DroneDeploy with:
- Cached patch generation (avoid reprocessing)
- Proper naming convention handling
- Memory-efficient loading
- Progress bars with tqdm
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
import hashlib
from typing import List, Tuple, Dict, Optional
import albumentations as A
from tqdm import tqdm
import pickle
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')


class DroneDeploy1024Dataset(Dataset):
    """
    DroneDeploy dataset with KDP-Net preprocessing methodology.
    
    Features:
    - Large images cropped to 1024×1024 patches (following research)
    - 6 classes optimized for UAV landing decisions
    - 6:2:2 train/val/test split as in KDP-Net paper
    - Edge enhancement preprocessing
    - Deterministic splits based on image hash
    """
    
    # DroneDeploy classes → Landing classes (research-validated)
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
        0: "ground",       # Safe flat landing (roads, dirt, pavement)
        1: "vegetation",   # Acceptable emergency landing (grass, trees)
        2: "building",     # Hard obstacles to avoid
        3: "water",        # Critical hazard - no landing
        4: "car",          # Dynamic obstacles
        5: "clutter"       # Mixed debris/objects, unknown areas
    }
    
    LANDING_SAFETY = {
        0: "SAFE_PRIMARY",    # ground → ideal landing
        1: "SAFE_SECONDARY",  # vegetation → acceptable emergency landing
        2: "AVOID",           # building → obstacle
        3: "CRITICAL_AVOID",  # water → hazard
        4: "AVOID",           # car → dynamic obstacle
        5: "CAUTION"          # clutter → requires analysis
    }
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        patch_size: int = 512,  # Smaller for RTX 4060 Ti
        stride_factor: float = 0.5,
        min_valid_pixels: float = 0.1,
        augmentation: bool = True,
        edge_enhancement: bool = False,
        cache_patches: bool = True,  # Enable caching
        force_preload: bool = False  # PERFORMANCE FIX: Force preloading option
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_factor)
        self.min_valid_pixels = min_valid_pixels
        self.augmentation = augmentation
        self.edge_enhancement = edge_enhancement
        self.cache_patches = cache_patches
        self.force_preload = force_preload
        
        # Paths
        self.images_dir = self.data_root / 'images'
        self.labels_dir = self.data_root / 'labels'
        self.cache_dir = self.data_root / 'patch_cache'
        
        if self.cache_patches:
            self.cache_dir.mkdir(exist_ok=True)
        
        print(f"🚁 Loading DroneDeploy dataset for {split}...")
        
        # OPTIMIZATION: Create fast lookup table for class mapping
        # Initialize with clutter (class 5) instead of ground (class 0) for unmapped values
        self.mapping_lut = np.full(256, 5, dtype=np.uint8)  # Default to clutter
        for dd_class, landing_class in self.DRONEDEPLOY_TO_LANDING.items():
            if dd_class < 256:  # Ensure valid index
                self.mapping_lut[dd_class] = landing_class
        
        # Check for cached patches first
        cache_file = self._get_cache_file()
        if self.cache_patches and cache_file.exists():
            print(f"   📂 Loading cached patches: {cache_file.name}")
            self.patches = self._load_cached_patches(cache_file)
        else:
            # Generate patches and cache them
            self.patches = self._generate_and_cache_patches()
        
        # Filter patches by split
        self.patches = self._filter_by_split()
        
        # Create transforms
        self.transform = self._setup_augmentation()
        
        # Analyze class distribution
        self._analyze_distribution()
        
        # Batch cache for on-demand loading
        self._batch_cache = {}
        self._max_cached_batches = max(50, len(self.patches) // 100)  # PERFORMANCE FIX: Much larger cache
        
        # PERFORMANCE FIX: Add option to preload all data for training
        should_preload = (
            self.force_preload or 
            (self.split == 'train' and len(self.patches) < 10000)  # Increased threshold
        )
        
        if should_preload:
            print(f"   🚀 PERFORMANCE MODE: Pre-loading all training patches into memory...")
            self._preload_all_patches()
        
        print(f"   Split {split}: {len(self.patches)} patches")
        print(f"   Batch cache size: {self._max_cached_batches}")
        print(f"   Preloaded: {hasattr(self, '_preloaded_patches')}")
    
    def _preload_all_patches(self):
        """Pre-load all patches into memory for maximum training speed."""
        self._preloaded_patches = {}
        
        print(f"   Loading {len(self.patches)} patches...")
        for i, patch_metadata in enumerate(tqdm(self.patches, desc="Pre-loading")):
            if 'batch_file' in patch_metadata:
                batch_file = patch_metadata['batch_file']
                patch_id = patch_metadata['patch_id']
                
                # Load batch if not cached
                if batch_file not in self._batch_cache:
                    self._load_batch(batch_file)
                
                # Find and store the actual patch data
                batch_patches = self._batch_cache[batch_file]
                for patch in batch_patches:
                    if patch['patch_id'] == patch_id:
                        self._preloaded_patches[patch_id] = {
                            'image_patch': patch['image_patch'].copy(),
                            'label_patch': patch['label_patch'].copy(),
                            'edge_map': patch.get('edge_map', None)
                        }
                        break
        
        print(f"   ✅ Pre-loaded {len(self._preloaded_patches)} patches into memory")
        # Clear batch cache to save memory since we have preloaded data
        self._batch_cache.clear()
    
    def _get_cache_file(self) -> Path:
        """Get cache file path based on parameters."""
        
        # Create hash of parameters for cache filename
        params = {
            'patch_size': self.patch_size,
            'stride_factor': self.stride / self.patch_size,
            'min_valid_pixels': self.min_valid_pixels,
            'edge_enhancement': self.edge_enhancement
        }
        
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return self.cache_dir / f"patches_{params_hash}.pkl"
    
    def _load_cached_patches(self, cache_file: Path) -> List[Dict]:
        """Load patches from cache with error handling."""
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if this is new metadata-based cache
            if 'patch_metadata' in cached_data:
                # New metadata-based cache
                if cached_data.get('image_count', 0) != len(list(self.images_dir.glob("*.tif"))):
                    print(f"   ⚠️  Cache outdated, regenerating...")
                    return self._generate_and_cache_patches()
                
                patch_metadata = cached_data['patch_metadata']
                if not isinstance(patch_metadata, list) or len(patch_metadata) == 0:
                    print(f"   ⚠️  Empty or invalid patches, regenerating...")
                    return self._generate_and_cache_patches()
                
                print(f"   ✅ Loaded {len(patch_metadata)} cached patch metadata")
                return patch_metadata
            
            # Legacy cache format - regenerate
            else:
                print(f"   ⚠️  Legacy cache format, regenerating...")
                return self._generate_and_cache_patches()
            
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"   ⚠️  Cache corrupted ({e}), regenerating...")
            # Remove corrupted cache file
            if cache_file.exists():
                cache_file.unlink()
            return self._generate_and_cache_patches()
        except Exception as e:
            print(f"   ❌ Unexpected cache error ({e}), regenerating...")
            if cache_file.exists():
                cache_file.unlink()
            return self._generate_and_cache_patches()
    
    def _generate_and_cache_patches(self) -> List[Dict]:
        """Generate patches and save to cache incrementally to avoid memory issues."""
        
        # Load image files
        image_files = self._load_image_files()
        
        print(f"   Found {len(image_files)} source images")
        
        # Create cache directory structure
        patches_dir = self.cache_dir / "patches"
        patches_dir.mkdir(exist_ok=True)
        
        # Clear any existing patch files
        for patch_file in patches_dir.glob("batch_*.pkl"):
            patch_file.unlink()
        
        # Track patch metadata (lightweight)
        all_patch_metadata = []
        batch_size = 5  # Process 5 images at a time to control memory
        batch_num = 0
        
        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="   Processing image batches"):
            batch_files = image_files[i:i + batch_size]
            batch_patches = []
            
            for img_path, label_path in batch_files:
                try:
                    # Load images
                    image = cv2.imread(str(img_path))
                    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                    
                    if image is None or label is None:
                        print(f"   ⚠️  Failed to load: {img_path.name}")
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w = image.shape[:2]
                    
                    # Generate patches from this image
                    image_patches = self._extract_patches_from_image(
                        image, label, img_path.stem, h, w
                    )
                    
                    batch_patches.extend(image_patches)
                    
                    # Add lightweight metadata for each patch
                    for patch_info in image_patches:
                        metadata = {
                            'source_image': patch_info['source_image'],
                            'source_label': patch_info['source_label'],
                            'coordinates': patch_info['coordinates'],
                            'patch_id': patch_info['patch_id'],
                            'batch_file': f"batch_{batch_num}.pkl"
                        }
                        all_patch_metadata.append(metadata)
                    
                    # Clear image data from memory immediately
                    del image, label, image_patches
                    
                except Exception as e:
                    print(f"   ❌ Error processing {img_path.name}: {e}")
                    continue
            
            # Save this batch to disk
            if batch_patches:
                batch_file = patches_dir / f"batch_{batch_num}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_patches, f)
                
                print(f"   💾 Saved batch {batch_num}: {len(batch_patches)} patches")
                
                # Clear batch from memory
                del batch_patches
                batch_num += 1
            
            # Force garbage collection
            import gc
            gc.collect()
        
        print(f"   Generated {len(all_patch_metadata)} patches in {batch_num} batches")
        
        # Save master metadata file
        cache_file = self._get_cache_file()
        cache_data = {
            'patch_metadata': all_patch_metadata,
            'image_count': len(image_files),
            'total_patches': len(all_patch_metadata),
            'num_batches': batch_num,
            'generated_at': str(Path.cwd()),
            'parameters': {
                'patch_size': self.patch_size,
                'stride_factor': self.stride / self.patch_size,
                'min_valid_pixels': self.min_valid_pixels,
                'edge_enhancement': self.edge_enhancement
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"   💾 Cached metadata: {cache_file.name}")
        
        return all_patch_metadata
    
    def _load_image_files(self) -> List[Tuple[Path, Path]]:
        """Load matching image and label file pairs."""
        
        image_files = []
        
        # Find all .tif images
        for img_path in self.images_dir.glob("*.tif"):
            # Handle naming pattern: image_name-ortho.tif → image_name-label.png
            img_stem = img_path.stem  # Remove .tif extension
            
            if img_stem.endswith('-ortho'):
                # Remove -ortho suffix and add -label
                base_name = img_stem[:-6]  # Remove '-ortho'
                label_name = f"{base_name}-label.png"
            else:
                # Fallback: just replace extension
                label_name = f"{img_stem}-label.png"
            
            label_path = self.labels_dir / label_name
            
            if label_path.exists():
                # Check file sizes (skip corrupted files)
                try:
                    img_size = img_path.stat().st_size
                    label_size = label_path.stat().st_size
                    
                    if img_size > 1024 * 1024 and label_size > 1024:  # Reasonable size check
                        image_files.append((img_path, label_path))
                        print(f"    Matched: {img_path.name} → {label_name}")
                    else:
                        print(f"   ⚠️  Skipping small file: {img_path.name}")
                        
                except Exception as e:
                    print(f"   ⚠️  Error checking {img_path.name}: {e}")
            else:
                print(f"   ⚠️  No label found for: {img_path.name} (expected: {label_name})")
        
        return sorted(image_files)
    
    def _extract_patches_from_image(self, image: np.ndarray, label: np.ndarray, img_stem: str, h: int, w: int) -> List[Dict]:
        """Extract 1024×1024 patches from a single image."""
        
        patches_from_image = 0
        patches_list = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Extract patch
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                label_patch = label[y:y+self.patch_size, x:x+self.patch_size]
                
                # Convert DroneDeploy labels to landing classes
                landing_patch = self._map_to_landing_classes(label_patch)
                
                # Validate patch quality
                if self._is_valid_patch(landing_patch):
                    # Create patch info
                    patch_info = {
                        'source_image': str(self.images_dir / f"{img_stem}-ortho.tif"), # Use ortho image for source
                        'source_label': str(self.labels_dir / f"{img_stem}-label.png"),
                        'coordinates': (x, y),
                        'image_patch': img_patch,
                        'label_patch': landing_patch,
                        'patch_id': f"{img_stem}_{x}_{y}"
                    }
                    
                    # Add edge map if enabled
                    if self.edge_enhancement:
                        edge_map = self._generate_edge_map(landing_patch)
                        patch_info['edge_map'] = edge_map
                    
                    patches_list.append(patch_info)
                    patches_from_image += 1
        
        print(f"      Generated: {patches_from_image} patches")
        return patches_list
    
    def _map_to_landing_classes(self, dronedeploy_label: np.ndarray) -> np.ndarray:
        """Map DroneDeploy classes to 6 landing classes using fast vectorized lookup."""
        # OPTIMIZATION: Use vectorized lookup table - orders of magnitude faster
        # Clip values to valid range to prevent index errors
        clipped_label = np.clip(dronedeploy_label, 0, 255)
        return self.mapping_lut[clipped_label]
    
    def _is_valid_patch(self, label_patch: np.ndarray) -> bool:
        """Check if patch has sufficient valid content."""
        
        # Count non-clutter pixels (avoid patches that are mostly unknown)
        valid_pixels = np.sum(label_patch != 5)  # Not clutter
        total_pixels = label_patch.size
        valid_ratio = valid_pixels / total_pixels
        
        # Require minimum valid content
        if valid_ratio < self.min_valid_pixels:
            return False
        
        # Prefer patches with interesting content (multiple classes)
        unique_classes = len(np.unique(label_patch))
        if unique_classes < 2:
            return False
        
        # Prioritize patches with safety-critical classes
        safety_critical = np.any([
            np.any(label_patch == 3),  # Water (critical hazard)
            np.any(label_patch == 4),  # Car (dynamic obstacle)
            np.any(label_patch == 2)   # Building (obstacle)
        ])
        
        return True
    
    def _generate_edge_map(self, label: np.ndarray) -> np.ndarray:
        """Generate edge map using Canny edge detection (KDP-Net methodology)."""
        
        # Convert to uint8 for Canny
        label_uint8 = label.astype(np.uint8) * 40  # Scale for better edge detection
        
        # Apply Canny edge detection
        edges = cv2.Canny(label_uint8, 50, 150)
        
        # Normalize to [0, 1]
        edge_map = edges.astype(np.float32) / 255.0
        
        return edge_map
    
    def _filter_by_split(self) -> List[Dict]:
        """Apply 6:2:2 train/val/test split deterministically."""
        
        # Group patches by source image for consistent splitting
        patches_by_source = {}
        for patch in self.patches:
            source = patch['source_image']
            if source not in patches_by_source:
                patches_by_source[source] = []
            patches_by_source[source].append(patch)
        
        # Sort source images deterministically
        source_images = sorted(patches_by_source.keys())
        
        # Create deterministic split based on hash
        train_sources, val_sources, test_sources = [], [], []
        
        for source in source_images:
            # Use hash for deterministic assignment
            hash_val = int(hashlib.md5(source.encode()).hexdigest(), 16)
            split_val = hash_val % 10
            
            if split_val < 6:  # 60% train
                train_sources.append(source)
            elif split_val < 8:  # 20% val
                val_sources.append(source)
            else:  # 20% test
                test_sources.append(source)
        
        # Select patches based on split
        if self.split == 'train':
            selected_sources = train_sources
        elif self.split == 'val':
            selected_sources = val_sources
        else:  # test
            selected_sources = test_sources
        
        # Collect patches from selected sources
        selected_patches = []
        for source in selected_sources:
            selected_patches.extend(patches_by_source[source])
        
        print(f"   Split distribution: train={len(train_sources)}, val={len(val_sources)}, test={len(test_sources)} source images")
        
        return selected_patches
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup augmentation pipeline."""
        
        transforms = [
            A.Resize(self.patch_size, self.patch_size, always_apply=True),
        ]
        
        if self.augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.5),  # Fixed: was A.Flip
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
            # ToTensorV2(), # Moved to __getitem__
        ])
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'}
        )
    
    def _analyze_distribution(self):
        """Analyze class distribution in the dataset."""
        
        if len(self.patches) == 0:
            print("⚠️  No patches available for distribution analysis")
            return
        
        # Sample a subset for analysis to avoid loading too much data
        max_analysis_samples = min(100, len(self.patches))
        sample_indices = np.random.choice(len(self.patches), max_analysis_samples, replace=False)
        
        class_counts = Counter()
        total_patches_analyzed = 0
        
        print(f"\n📊 Class Distribution ({max_analysis_samples} patches sampled):")
        
        for idx in sample_indices:
            try:
                patch_metadata = self.patches[idx]
                
                # Load actual patch data
                if 'batch_file' in patch_metadata:
                    # New metadata format
                    batch_file = patch_metadata['batch_file']
                    patch_id = patch_metadata['patch_id']
                    
                    # Load batch if not cached
                    if batch_file not in self._batch_cache:
                        self._load_batch(batch_file)
                    
                    # Find the actual patch data
                    batch_patches = self._batch_cache[batch_file]
                    actual_patch = None
                    for patch in batch_patches:
                        if patch['patch_id'] == patch_id:
                            actual_patch = patch
                            break
                    
                    if actual_patch is None:
                        continue
                    
                    label_patch = actual_patch['label_patch']
                else:
                    # Legacy format
                    label_patch = patch_metadata['label_patch']
                
                # Count pixels for each class
                unique, counts = np.unique(label_patch, return_counts=True)
                for class_id, count in zip(unique, counts):
                    class_counts[class_id] += count
                
                total_patches_analyzed += 1
                
            except Exception as e:
                continue  # Skip problematic patches
        
        if total_patches_analyzed == 0:
            print("⚠️  No patches could be analyzed")
            return
        
        # Calculate percentages and display
        total_pixels = sum(class_counts.values())
        
        for class_id in range(6):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
            class_name = self.DRONEDEPLOY_TO_LANDING.get(class_id, f"class_{class_id}")
            landing_category = self.LANDING_CLASSES.get(class_id, "UNKNOWN")
            
            print(f"   {class_id} ({class_name:<10}) [{landing_category:<15}]: {percentage:5.1f}% ({count:,} pixels)")
        
        print(f"   Total patches analyzed: {total_patches_analyzed}")
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a patch by index - loads patch data on-demand."""
        
        patch_metadata = self.patches[idx]
        patch_id = patch_metadata['patch_id']
        
        # PERFORMANCE FIX: Use preloaded data if available
        if hasattr(self, '_preloaded_patches') and patch_id in self._preloaded_patches:
            preloaded = self._preloaded_patches[patch_id]
            image = preloaded['image_patch'].copy()
            mask = preloaded['label_patch'].copy()
            edge_map = preloaded.get('edge_map')
        
        # Check if this is metadata (new format) or actual patch data (legacy)
        elif 'batch_file' in patch_metadata:
            # New metadata format - load actual patch data on-demand
            batch_file = patch_metadata['batch_file']
            
            # Load batch if not cached
            if batch_file not in self._batch_cache:
                self._load_batch(batch_file)
            
            # Find the actual patch data in the loaded batch
            batch_patches = self._batch_cache[batch_file]
            actual_patch = None
            for patch in batch_patches:
                if patch['patch_id'] == patch_id:
                    actual_patch = patch
                    break
            
            if actual_patch is None:
                raise ValueError(f"Patch {patch_id} not found in batch {batch_file}")
            
            # Get patches from actual patch data
            image = actual_patch['image_patch'].copy()
            mask = actual_patch['label_patch'].copy()
            edge_map = actual_patch.get('edge_map')
            
        else:
            # Legacy format - patch data is directly available
            image = patch_metadata['image_patch'].copy()
            mask = patch_metadata['label_patch'].copy()
            edge_map = patch_metadata.get('edge_map')
        
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
        
        result = {
            'image': image,
            'mask': mask,
            'patch_id': patch_metadata['patch_id']
        }
        
        # Add edge map if available
        if edge_map is not None:
            if not isinstance(edge_map, torch.Tensor):
                edge_map = torch.from_numpy(edge_map).float()
            result['edge_map'] = edge_map
        
        return result
    
    def _load_batch(self, batch_file: str):
        """Load a batch file into memory cache."""
        
        # Manage cache size - remove oldest batch if needed
        if len(self._batch_cache) >= self._max_cached_batches:
            # Remove the first (oldest) cached batch
            oldest_batch = next(iter(self._batch_cache))
            del self._batch_cache[oldest_batch]
        
        # Load the requested batch
        batch_path = self.cache_dir / "patches" / batch_file
        
        try:
            with open(batch_path, 'rb') as f:
                batch_data = pickle.load(f)
            
            self._batch_cache[batch_file] = batch_data
            
        except Exception as e:
            raise ValueError(f"Failed to load batch {batch_file}: {e}")
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        
        if len(self.patches) == 0:
            # Return uniform weights if no patches
            return torch.ones(6)
        
        # Sample distribution analysis from a subset of patches
        max_samples = min(100, len(self.patches))
        sample_indices = np.random.choice(len(self.patches), max_samples, replace=False)
        
        class_counts = Counter()
        
        for idx in sample_indices:
            try:
                patch_metadata = self.patches[idx]
                
                # Load actual patch data
                if 'batch_file' in patch_metadata:
                    # New metadata format - load on-demand
                    batch_file = patch_metadata['batch_file']
                    patch_id = patch_metadata['patch_id']
                    
                    # Load batch if not cached
                    if batch_file not in self._batch_cache:
                        self._load_batch(batch_file)
                    
                    # Find the actual patch data
                    batch_patches = self._batch_cache[batch_file]
                    actual_patch = None
                    for patch in batch_patches:
                        if patch['patch_id'] == patch_id:
                            actual_patch = patch
                            break
                    
                    if actual_patch is None:
                        continue
                    
                    label_patch = actual_patch['label_patch']
                else:
                    # Legacy format
                    label_patch = patch_metadata['label_patch']
                
                # Count classes in this patch
                unique, counts = np.unique(label_patch, return_counts=True)
                for class_id, count in zip(unique, counts):
                    class_counts[class_id] += count
                    
            except Exception as e:
                print(f"   ⚠️  Error processing patch for class weights: {e}")
                continue
        
        if not class_counts:
            # Fallback if no valid patches found
            return torch.ones(6)
        
        # Compute inverse frequency weights
        total_pixels = sum(class_counts.values())
        class_weights = torch.ones(6)
        
        for class_id in range(6):
            count = class_counts.get(class_id, 1)  # Avoid division by zero
            class_weights[class_id] = total_pixels / (6 * count)
        
        # Apply safety multipliers for UAV landing safety
        class_weights[3] *= 2.0  # Water (critical hazard)
        class_weights[4] *= 1.5  # Car (obstacle)
        
        # Normalize and clip extreme values
        class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
        
        return class_weights


def create_dronedeploy_datasets(
    data_root: str,
    patch_size: int = 1024,
    force_preload: bool = False,  # PERFORMANCE FIX
    **kwargs
) -> Dict[str, DroneDeploy1024Dataset]:
    """
    Create DroneDeploy datasets with KDP-Net methodology.
    
    Args:
        data_root: Root directory containing DroneDeploy data
        patch_size: Patch size (1024 following research)
        force_preload: Force preloading all data into memory for training speed
        **kwargs: Additional dataset parameters
        
    Returns:
        Dictionary with train/val/test datasets
    """
    
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = DroneDeploy1024Dataset(
                data_root=data_root,
                split=split,
                patch_size=patch_size,
                force_preload=force_preload and split == 'train',  # Only preload training data
                **kwargs
            )
            datasets[split] = dataset
            
        except Exception as e:
            print(f"❌ Failed to create {split} dataset: {e}")
            datasets[split] = None
    
    return datasets


if __name__ == "__main__":
    # Test dataset creation
    print("🚁 Testing DroneDeploy 1024 Dataset...")
    
    try:
        # Test with your data path
        datasets = create_dronedeploy_datasets(
            data_root="../datasets/drone_deploy_dataset_intermediate/dataset-medium",
            patch_size=1024,
            augmentation=True,
            edge_enhancement=True
        )
        
        print(f"\n Dataset creation successful!")
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
            
            if 'edge_map' in sample:
                print(f"   Edge map shape: {sample['edge_map'].shape}")
        
        # Test class weights
        if datasets['train'] is not None:
            class_weights = datasets['train'].get_class_weights()
            print(f"\n⚖️  Class weights: {class_weights}")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc() 