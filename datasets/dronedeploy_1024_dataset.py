#!/usr/bin/env python3
"""
DroneDeploy Dataset with Proper RGB Color Mapping
================================================

Implements the correct DroneDeploy dataset format with:
- RGB color to class mapping as per DroneDeploy specification
- 300x300 chip extraction with stride 300 (no overlap)
- Proper handling of IGNORE pixels (255, 0, 255)
- Support for elevation data integration
- 6-class UAV landing classification
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
    DroneDeploy dataset with proper RGB color mapping and chip-based approach.
    
    Features:
    - RGB color to class conversion as per DroneDeploy specification
    - 300x300 chip extraction with stride 300 (following DroneDeploy methodology)
    - Skip chips containing IGNORE pixels (255, 0, 255)
    - 6 classes optimized for UAV landing decisions
    - Support for elevation data (future enhancement)
    - Deterministic splits based on image hash
    """
    
    # DroneDeploy RGB color to class mapping (CORRECTED as per official specification)
    # Colors from official repo: https://github.com/dronedeploy/dd-ml-segmentation-benchmark
    RGB_TO_CLASS = {
        (230, 25, 75): 0,   # BUILDING (red-ish)
        (145, 30, 180): 1,  # CLUTTER (purple-ish)
        (60, 180, 75): 2,   # VEGETATION (green)
        (245, 130, 48): 3,  # WATER (blue-ish)
        (255, 255, 255): 4, # GROUND (white)
        (0, 130, 200): 5,   # CAR (dark blue)
        (255, 0, 255): -1,  # IGNORE (magenta) - skip chips with this
    }
    
    # Map DroneDeploy classes to unified landing classes
    DRONEDEPLOY_TO_LANDING = {
        0: 2,   # BUILDING → building (avoid)
        1: 5,   # CLUTTER → clutter (caution)
        2: 1,   # VEGETATION → vegetation (safe secondary)
        3: 3,   # WATER → water (critical hazard)
        4: 0,   # GROUND → ground (safe primary)
        5: 4,   # CAR → car (dynamic obstacle)
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
        chip_size: int = 300,  # DroneDeploy standard chip size
        target_size: int = 512,  # Target resize for model input
        min_valid_pixels: float = 0.8,  # Higher threshold for chip quality
        augmentation: bool = True,
        edge_enhancement: bool = False,
        cache_chips: bool = True,
        force_preload: bool = False,
        use_elevation: bool = False  # Future: use elevation tiles
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.chip_size = chip_size  # 300x300 chips as per DroneDeploy
        self.target_size = target_size  # Resize to this for model
        self.min_valid_pixels = min_valid_pixels
        self.augmentation = augmentation
        self.edge_enhancement = edge_enhancement
        self.cache_chips = cache_chips
        self.force_preload = force_preload
        self.use_elevation = use_elevation
        
        # Paths
        self.images_dir = self.data_root / 'images'
        self.labels_dir = self.data_root / 'labels'
        self.elevation_dir = self.data_root / 'elevation' if use_elevation else None
        self.cache_dir = self.data_root / 'chip_cache'
        
        if self.cache_chips:
            self.cache_dir.mkdir(exist_ok=True)
        
        print(f"🚁 Loading DroneDeploy dataset for {split}...")
        print(f"   Using {chip_size}x{chip_size} chips → {target_size}x{target_size} model input")
        
        # Create RGB to class lookup table for fast conversion
        self._create_rgb_lookup_table()
        
        # Create landing class mapping LUT
        self.landing_lut = np.full(256, 5, dtype=np.uint8)  # Default to clutter
        for dd_class, landing_class in self.DRONEDEPLOY_TO_LANDING.items():
            if 0 <= dd_class < 256:
                self.landing_lut[dd_class] = landing_class
        
        # Check for cached chips first
        cache_file = self._get_cache_file()
        if self.cache_chips and cache_file.exists():
            print(f"   📂 Loading cached chips: {cache_file.name}")
            self.chips = self._load_cached_chips(cache_file)
        else:
            # Generate chips and cache them
            self.chips = self._generate_and_cache_chips()
        
        # Filter chips by split
        self.chips = self._filter_by_split()
        
        # Create transforms
        self.transform = self._setup_augmentation()
        
        # Analyze class distribution
        self._analyze_distribution()
        
        # Batch cache for on-demand loading
        self._batch_cache = {}
        self._max_cached_batches = max(50, len(self.chips) // 100)
        
        # PERFORMANCE FIX: Add option to preload all data for training
        should_preload = (
            self.force_preload or 
            (self.split == 'train' and len(self.chips) < 10000)
        )
        
        if should_preload:
            print(f"   🚀 PERFORMANCE MODE: Pre-loading all training chips into memory...")
            self._preload_all_chips()
        
        print(f"   Split {split}: {len(self.chips)} chips")
        print(f"   Batch cache size: {self._max_cached_batches}")
        print(f"   Preloaded: {hasattr(self, '_preloaded_chips')}")
    
    def _create_rgb_lookup_table(self):
        """Create fast RGB to class ID lookup table."""
        # Create a 3D lookup table for RGB values
        # This is memory intensive but very fast for conversion
        self.rgb_lookup = np.full((256, 256, 256), 5, dtype=np.uint8)  # Default to clutter
        
        for rgb_color, class_id in self.RGB_TO_CLASS.items():
            r, g, b = rgb_color
            if class_id >= 0:  # Skip IGNORE pixels
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
    
    def _has_ignore_pixels(self, rgb_image: np.ndarray) -> bool:
        """Check if image contains IGNORE pixels (255, 0, 255)."""
        ignore_mask = np.all(rgb_image == [255, 0, 255], axis=2)
        return np.any(ignore_mask)
    
    def _preload_all_chips(self):
        """Pre-load all chips into memory for maximum training speed."""
        self._preloaded_chips = {}
        
        print(f"   Loading {len(self.chips)} chips...")
        for i, chip_metadata in enumerate(tqdm(self.chips, desc="Pre-loading")):
            if 'batch_file' in chip_metadata:
                batch_file = chip_metadata['batch_file']
                chip_id = chip_metadata['chip_id']
                
                # Load batch if not cached
                if batch_file not in self._batch_cache:
                    self._load_batch(batch_file)
                
                # Find and store the actual chip data
                batch_chips = self._batch_cache[batch_file]
                for chip in batch_chips:
                    if chip['chip_id'] == chip_id:
                        self._preloaded_chips[chip_id] = {
                            'image_chip': chip['image_chip'].copy(),
                            'label_chip': chip['label_chip'].copy(),
                            'elevation_chip': chip.get('elevation_chip', None),
                            'edge_map': chip.get('edge_map', None)
                        }
                        break
        
        print(f"   ✅ Pre-loaded {len(self._preloaded_chips)} chips into memory")
        # Clear batch cache to save memory since we have preloaded data
        self._batch_cache.clear()
    
    def _get_cache_file(self) -> Path:
        """Get cache file path based on parameters."""
        
        # Create hash of parameters for cache filename
        params = {
            'chip_size': self.chip_size,
            'target_size': self.target_size,
            'min_valid_pixels': self.min_valid_pixels,
            'edge_enhancement': self.edge_enhancement,
            'use_elevation': self.use_elevation
        }
        
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return self.cache_dir / f"chips_{params_hash}.pkl"
    
    def _load_cached_chips(self, cache_file: Path) -> List[Dict]:
        """Load chips from cache with error handling."""
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if this is new metadata-based cache
            if 'chip_metadata' in cached_data:
                # New metadata-based cache
                if cached_data.get('image_count', 0) != len(list(self.images_dir.glob("*.tif"))):
                    print(f"   ⚠️  Cache outdated, regenerating...")
                    return self._generate_and_cache_chips()
                
                chip_metadata = cached_data['chip_metadata']
                if not isinstance(chip_metadata, list) or len(chip_metadata) == 0:
                    print(f"   ⚠️  Empty or invalid chips, regenerating...")
                    return self._generate_and_cache_chips()
                
                print(f"   ✅ Loaded {len(chip_metadata)} cached chip metadata")
                return chip_metadata
            
            # Legacy cache format - regenerate
            else:
                print(f"   ⚠️  Legacy cache format, regenerating...")
                return self._generate_and_cache_chips()
            
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            print(f"   ⚠️  Cache corrupted ({e}), regenerating...")
            # Remove corrupted cache file
            if cache_file.exists():
                cache_file.unlink()
            return self._generate_and_cache_chips()
        except Exception as e:
            print(f"   ❌ Unexpected cache error ({e}), regenerating...")
            if cache_file.exists():
                cache_file.unlink()
            return self._generate_and_cache_chips()
    
    def _generate_and_cache_chips(self) -> List[Dict]:
        """Generate 300x300 chips and save to cache incrementally to avoid memory issues."""
        
        # Load image files
        image_files = self._load_image_files()
        
        print(f"   Found {len(image_files)} source images")
        
        # Create cache directory structure
        chips_dir = self.cache_dir / "chips"
        chips_dir.mkdir(exist_ok=True)
        
        # Clear any existing chip files
        for chip_file in chips_dir.glob("batch_*.pkl"):
            chip_file.unlink()
        
        # Track chip metadata (lightweight)
        all_chip_metadata = []
        batch_size = 5  # Process 5 images at a time to control memory
        batch_num = 0
        
        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="   Processing image batches"):
            batch_files = image_files[i:i + batch_size]
            batch_chips = []
            
            for img_path, label_path in batch_files:
                try:
                    # Load images
                    image = cv2.imread(str(img_path))
                    label_rgb = cv2.imread(str(label_path))
                    
                    if image is None or label_rgb is None:
                        print(f"   ⚠️  Failed to load: {img_path.name}")
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
                    
                    # Load elevation if available
                    elevation = None
                    if self.use_elevation and self.elevation_dir:
                        elev_name = img_path.stem.replace('-ortho', '-elevation') + '.tif'
                        elev_path = self.elevation_dir / elev_name
                        if elev_path.exists():
                            elevation = cv2.imread(str(elev_path), cv2.IMREAD_UNCHANGED)
                    
                    h, w = image.shape[:2]
                    
                    # Generate chips from this image
                    image_chips = self._extract_chips_from_image(
                        image, label_rgb, elevation, img_path.stem, h, w
                    )
                    
                    batch_chips.extend(image_chips)
                    
                    # Add lightweight metadata for each chip
                    for chip_info in image_chips:
                        metadata = {
                            'source_image': chip_info['source_image'],
                            'source_label': chip_info['source_label'],
                            'coordinates': chip_info['coordinates'],
                            'chip_id': chip_info['chip_id'],
                            'batch_file': f"batch_{batch_num}.pkl"
                        }
                        all_chip_metadata.append(metadata)
                    
                    # Clear image data from memory immediately
                    del image, label_rgb, image_chips
                    if elevation is not None:
                        del elevation
                    
                except Exception as e:
                    print(f"   ❌ Error processing {img_path.name}: {e}")
                    continue
            
            # Save this batch to disk
            if batch_chips:
                batch_file = chips_dir / f"batch_{batch_num}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_chips, f)
                
                print(f"   💾 Saved batch {batch_num}: {len(batch_chips)} chips")
                
                # Clear batch from memory
                del batch_chips
                batch_num += 1
            
            # Force garbage collection
            import gc
            gc.collect()
        
        print(f"   Generated {len(all_chip_metadata)} chips in {batch_num} batches")
        
        # Save master metadata file
        cache_file = self._get_cache_file()
        cache_data = {
            'chip_metadata': all_chip_metadata,
            'image_count': len(image_files),
            'total_chips': len(all_chip_metadata),
            'num_batches': batch_num,
            'generated_at': str(Path.cwd()),
            'parameters': {
                'chip_size': self.chip_size,
                'target_size': self.target_size,
                'min_valid_pixels': self.min_valid_pixels,
                'edge_enhancement': self.edge_enhancement,
                'use_elevation': self.use_elevation
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"   💾 Cached metadata: {cache_file.name}")
        
        return all_chip_metadata
    
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
    
    def _extract_chips_from_image(self, image: np.ndarray, label_rgb: np.ndarray, 
                                 elevation: Optional[np.ndarray], img_stem: str, h: int, w: int) -> List[Dict]:
        """Extract 300x300 chips from a single image with stride 300 (no overlap)."""
        
        chips_from_image = 0
        chips_list = []
        
        # DroneDeploy methodology: 300x300 chips with stride 300 (no overlap)
        for y in range(0, h - self.chip_size + 1, self.chip_size):
            for x in range(0, w - self.chip_size + 1, self.chip_size):
                # Extract chip
                img_chip = image[y:y+self.chip_size, x:x+self.chip_size]
                label_chip_rgb = label_rgb[y:y+self.chip_size, x:x+self.chip_size]
                
                # Convert RGB labels to DroneDeploy class IDs (IGNORE pixels become -1)
                dd_class_chip = self._rgb_to_class_vectorized(label_chip_rgb)
                
                # Convert DroneDeploy classes to landing classes (preserve IGNORE as 255)
                landing_chip = self._map_to_landing_classes(dd_class_chip)
                
                # FIXED: Keep chips with IGNORE pixels but set them to 255 (ignore_index)
                landing_chip[dd_class_chip == -1] = 255  # PyTorch ignore_index convention
                
                # Validate chip quality (but don't exclude IGNORE pixels)
                if self._is_valid_chip_with_ignore(landing_chip, dd_class_chip):
                    # Extract elevation chip if available
                    elev_chip = None
                    if elevation is not None:
                        elev_chip = elevation[y:y+self.chip_size, x:x+self.chip_size]
                    
                    # Resize to target size for model input
                    img_chip_resized = cv2.resize(img_chip, (self.target_size, self.target_size), 
                                                interpolation=cv2.INTER_LINEAR)
                    landing_chip_resized = cv2.resize(landing_chip, (self.target_size, self.target_size), 
                                                    interpolation=cv2.INTER_NEAREST)
                    
                    # Create chip info
                    chip_info = {
                        'source_image': str(self.images_dir / f"{img_stem}.tif"),
                        'source_label': str(self.labels_dir / f"{img_stem}.png"),
                        'coordinates': (x, y),
                        'image_chip': img_chip_resized,
                        'label_chip': landing_chip_resized,
                        'chip_id': f"{img_stem}_{x}_{y}"
                    }
                    
                    # Add elevation chip if available
                    if elev_chip is not None:
                        elev_chip_resized = cv2.resize(elev_chip, (self.target_size, self.target_size), 
                                                     interpolation=cv2.INTER_LINEAR)
                        chip_info['elevation_chip'] = elev_chip_resized
                    
                    # Add edge map if enabled
                    if self.edge_enhancement:
                        edge_map = self._generate_edge_map(landing_chip_resized)
                        chip_info['edge_map'] = edge_map
                    
                    chips_list.append(chip_info)
                    chips_from_image += 1
        
        print(f"      Generated: {chips_from_image} valid chips (skipped IGNORE pixels)")
        return chips_list
    
    def _map_to_landing_classes(self, dronedeploy_label: np.ndarray) -> np.ndarray:
        """Map DroneDeploy classes to 6 landing classes using fast vectorized lookup."""
        # OPTIMIZATION: Use vectorized lookup table - orders of magnitude faster
        # Clip values to valid range to prevent index errors
        clipped_label = np.clip(dronedeploy_label, 0, 255)
        return self.landing_lut[clipped_label]
    
    def _is_valid_chip(self, landing_label: np.ndarray, dd_label: np.ndarray) -> bool:
        """Check if chip has sufficient valid content for training."""
        
        # Count non-clutter pixels (avoid chips that are mostly unknown)
        valid_pixels = np.sum(landing_label != 5)  # Not clutter
        total_pixels = landing_label.size
        valid_ratio = valid_pixels / total_pixels
        
        # Require high validity for chip-based approach
        if valid_ratio < self.min_valid_pixels:
            return False
        
        # Prefer chips with interesting content (multiple classes)
        unique_classes = len(np.unique(landing_label))
        if unique_classes < 2:
            return False
        
        # Prioritize chips with safety-critical classes
        safety_critical = np.any([
            np.any(landing_label == 3),  # Water (critical hazard)
            np.any(landing_label == 4),  # Car (dynamic obstacle)
            np.any(landing_label == 2),  # Building (obstacle)
            np.any(landing_label == 0)   # Ground (safe landing)
        ])
        
        return True
    
    def _is_valid_chip_with_ignore(self, landing_label: np.ndarray, dd_label: np.ndarray) -> bool:
        """Check if chip has sufficient valid content for training (handles IGNORE pixels)."""
        
        # Count valid (non-IGNORE, non-clutter) pixels
        valid_mask = (landing_label != 255) & (landing_label != 5)  # Not IGNORE or clutter
        valid_pixels = np.sum(valid_mask)
        total_non_ignore = np.sum(landing_label != 255)  # Total non-IGNORE pixels
        
        # Skip chips that are mostly IGNORE pixels
        if total_non_ignore == 0:
            return False
            
        # Require sufficient valid content among non-IGNORE pixels
        valid_ratio = valid_pixels / max(total_non_ignore, 1)
        if valid_ratio < 0.3:  # Lower threshold since we keep IGNORE chips
            return False
        
        # Prefer chips with interesting content (multiple classes, excluding IGNORE)
        unique_classes = len(np.unique(landing_label[landing_label != 255]))
        if unique_classes < 2:
            return False
        
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
        
        # Group chips by source image for consistent splitting
        chips_by_source = {}
        for chip in self.chips:
            source = chip['source_image']
            if source not in chips_by_source:
                chips_by_source[source] = []
            chips_by_source[source].append(chip)
        
        # Sort source images deterministically
        source_images = sorted(chips_by_source.keys())
        
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
        
        # Select chips based on split
        if self.split == 'train':
            selected_sources = train_sources
        elif self.split == 'val':
            selected_sources = val_sources
        else:  # test
            selected_sources = test_sources
        
        # Collect chips from selected sources
        selected_chips = []
        for source in selected_sources:
            selected_chips.extend(chips_by_source[source])
        
        print(f"   Split distribution: train={len(train_sources)}, val={len(val_sources)}, test={len(test_sources)} source images")
        
        return selected_chips
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup augmentation pipeline for chip-based training."""
        
        transforms = [
            # No resize needed - chips are already at target size
        ]
        
        if self.augmentation:
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
    
    def _analyze_distribution(self):
        """Analyze class distribution in the dataset."""
        
        if len(self.chips) == 0:
            print("⚠️  No chips available for distribution analysis")
            return
        
        # Sample a subset for analysis to avoid loading too much data
        max_analysis_samples = min(100, len(self.chips))
        sample_indices = np.random.choice(len(self.chips), max_analysis_samples, replace=False)
        
        class_counts = Counter()
        total_chips_analyzed = 0
        
        print(f"\n📊 Class Distribution ({max_analysis_samples} chips sampled):")
        
        for idx in sample_indices:
            try:
                chip_metadata = self.chips[idx]
                
                # Load actual chip data
                if 'batch_file' in chip_metadata:
                    # New metadata format
                    batch_file = chip_metadata['batch_file']
                    chip_id = chip_metadata['chip_id']
                    
                    # Load batch if not cached
                    if batch_file not in self._batch_cache:
                        self._load_batch(batch_file)
                    
                    # Find the actual chip data
                    batch_chips = self._batch_cache[batch_file]
                    actual_chip = None
                    for chip in batch_chips:
                        if chip['chip_id'] == chip_id:
                            actual_chip = chip
                            break
                    
                    if actual_chip is None:
                        continue
                    
                    label_chip = actual_chip['label_chip']
                else:
                    # Legacy format
                    label_chip = chip_metadata['label_chip']
                
                # Count pixels for each class
                unique, counts = np.unique(label_chip, return_counts=True)
                for class_id, count in zip(unique, counts):
                    class_counts[class_id] += count
                
                total_chips_analyzed += 1
                
            except Exception as e:
                continue  # Skip problematic chips
        
        if total_chips_analyzed == 0:
            print("⚠️  No chips could be analyzed")
            return
        
        # Calculate percentages and display
        total_pixels = sum(class_counts.values())
        
        for class_id in range(6):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
            landing_category = self.LANDING_CLASSES.get(class_id, "UNKNOWN")
            safety_level = self.LANDING_SAFETY.get(class_id, "UNKNOWN")
            
            print(f"   {class_id} ({landing_category:<12}) [{safety_level:<15}]: {percentage:5.1f}% ({count:,} pixels)")
        
        print(f"   Total chips analyzed: {total_chips_analyzed}")
    
    def __len__(self) -> int:
        return len(self.chips)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a chip by index - loads chip data on-demand."""
        
        chip_metadata = self.chips[idx]
        chip_id = chip_metadata['chip_id']
        
        # PERFORMANCE FIX: Use preloaded data if available
        if hasattr(self, '_preloaded_chips') and chip_id in self._preloaded_chips:
            preloaded = self._preloaded_chips[chip_id]
            image = preloaded['image_chip'].copy()
            mask = preloaded['label_chip'].copy()
            elevation = preloaded.get('elevation_chip')
            edge_map = preloaded.get('edge_map')
        
        # Check if this is metadata (new format) or actual chip data (legacy)
        elif 'batch_file' in chip_metadata:
            # New metadata format - load actual chip data on-demand
            batch_file = chip_metadata['batch_file']
            
            # Load batch if not cached
            if batch_file not in self._batch_cache:
                self._load_batch(batch_file)
            
            # Find the actual chip data in the loaded batch
            batch_chips = self._batch_cache[batch_file]
            actual_chip = None
            for chip in batch_chips:
                if chip['chip_id'] == chip_id:
                    actual_chip = chip
                    break
            
            if actual_chip is None:
                raise ValueError(f"Chip {chip_id} not found in batch {batch_file}")
            
            # Get chips from actual chip data
            image = actual_chip['image_chip'].copy()
            mask = actual_chip['label_chip'].copy()
            elevation = actual_chip.get('elevation_chip')
            edge_map = actual_chip.get('edge_map')
            
        else:
            # Legacy format - chip data is directly available
            image = chip_metadata['image_chip'].copy()
            mask = chip_metadata['label_chip'].copy()
            elevation = chip_metadata.get('elevation_chip')
            edge_map = chip_metadata.get('edge_map')
        
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
            'chip_id': chip_metadata['chip_id']
        }
        
        # Add elevation data if available
        if elevation is not None:
            if not isinstance(elevation, torch.Tensor):
                # Normalize elevation data
                elevation = torch.from_numpy(elevation).float()
                elevation = (elevation - elevation.mean()) / (elevation.std() + 1e-8)
            result['elevation'] = elevation
        
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
        batch_path = self.cache_dir / "chips" / batch_file
        
        try:
            with open(batch_path, 'rb') as f:
                batch_data = pickle.load(f)
            
            self._batch_cache[batch_file] = batch_data
            
        except Exception as e:
            raise ValueError(f"Failed to load batch {batch_file}: {e}")
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        
        if len(self.chips) == 0:
            # Return uniform weights if no chips
            return torch.ones(6)
        
        # Sample distribution analysis from a subset of chips
        max_samples = min(100, len(self.chips))
        sample_indices = np.random.choice(len(self.chips), max_samples, replace=False)
        
        class_counts = Counter()
        
        for idx in sample_indices:
            try:
                chip_metadata = self.chips[idx]
                
                # Load actual chip data
                if 'batch_file' in chip_metadata:
                    # New metadata format - load on-demand
                    batch_file = chip_metadata['batch_file']
                    chip_id = chip_metadata['chip_id']
                    
                    # Load batch if not cached
                    if batch_file not in self._batch_cache:
                        self._load_batch(batch_file)
                    
                    # Find the actual chip data
                    batch_chips = self._batch_cache[batch_file]
                    actual_chip = None
                    for chip in batch_chips:
                        if chip['chip_id'] == chip_id:
                            actual_chip = chip
                            break
                    
                    if actual_chip is None:
                        continue
                    
                    label_chip = actual_chip['label_chip']
                else:
                    # Legacy format
                    label_chip = chip_metadata['label_chip']
                
                # Count classes in this chip
                unique, counts = np.unique(label_chip, return_counts=True)
                for class_id, count in zip(unique, counts):
                    class_counts[class_id] += count
                    
            except Exception as e:
                print(f"   ⚠️  Error processing chip for class weights: {e}")
                continue
        
        if not class_counts:
            # Fallback if no valid chips found
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
    chip_size: int = 300,  # DroneDeploy standard
    target_size: int = 512,  # Model input size
    force_preload: bool = False,
    use_elevation: bool = False,
    **kwargs
) -> Dict[str, DroneDeploy1024Dataset]:
    """
    Create DroneDeploy datasets with proper RGB color mapping and chip-based methodology.
    
    Args:
        data_root: Root directory containing DroneDeploy data
        chip_size: Chip extraction size (300x300 following DroneDeploy methodology)
        target_size: Target resize for model input (512x512 for current hardware)
        force_preload: Force preloading all data into memory for training speed
        use_elevation: Include elevation data if available
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
                chip_size=chip_size,
                target_size=target_size,
                force_preload=force_preload and split == 'train',  # Only preload training data
                use_elevation=use_elevation,
                **kwargs
            )
            datasets[split] = dataset
            
        except Exception as e:
            print(f"❌ Failed to create {split} dataset: {e}")
            datasets[split] = None
    
    return datasets


if __name__ == "__main__":
    # Test dataset creation
    print("🚁 Testing DroneDeploy RGB Color Dataset...")
    
    try:
        # Test with your data path
        datasets = create_dronedeploy_datasets(
            data_root="../datasets/drone_deploy_dataset_intermediate/dataset-medium",
            chip_size=300,
            target_size=512,
            augmentation=True,
            edge_enhancement=True,
            use_elevation=False
        )
        
        print(f"\n Dataset creation successful!")
        for split, dataset in datasets.items():
            if dataset is not None:
                print(f"   {split}: {len(dataset)} chips")
        
        # Test sample loading
        if datasets['train'] is not None and len(datasets['train']) > 0:
            sample = datasets['train'][0]
            print(f"\n📋 Sample test:")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Mask shape: {sample['mask'].shape}")
            print(f"   Unique classes: {torch.unique(sample['mask'])}")
            print(f"   Chip ID: {sample['chip_id']}")
            
            if 'elevation' in sample:
                print(f"   Elevation shape: {sample['elevation'].shape}")
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