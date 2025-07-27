#!/usr/bin/env python3
"""
DroneDeploy HDF5 Dataset
========================

High-performance dataset using HDF5 for optimal storage and access:
- Single file storage (easy deployment/backup)
- Built-in compression (smaller storage)
- Random access (no loading everything)
- Optimized chunking (faster I/O)
- Cross-platform compatibility

This is the optimal long-term solution for large patch datasets.

Storage Structure:
    dronedeploy_patches.h5
    ├── images/
    │   ├── train          # (N, H, W, 3) compressed patches
    │   ├── val            # (N, H, W, 3) compressed patches
    │   └── test           # (N, H, W, 3) compressed patches
    ├── masks/
    │   ├── train          # (N, H, W) compressed masks
    │   ├── val            # (N, H, W) compressed masks
    │   └── test           # (N, H, W) compressed masks
    └── metadata/
        ├── patch_ids      # Patch identifier strings
        ├── class_mapping  # DroneDeploy → Landing class mapping
        └── dataset_info   # Dataset statistics and metadata

Usage:
    # Convert existing data to HDF5 (run once)
    python dronedeploy_hdf5_dataset.py --convert /path/to/dronedeploy/data
    
    # Use for training (much faster!)
    dataset = DroneDeployHDF5Dataset('dronedeploy_patches.h5', split='train')
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
from pathlib import Path
import json
import albumentations as A
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Union
import argparse
import hashlib
import warnings
warnings.filterwarnings('ignore')


class DroneDeployHDF5Dataset(Dataset):
    """
    Ultra-fast DroneDeploy dataset using HDF5 storage.
    
    Benefits:
    - Single file (easy deployment)
    - Compressed storage (smaller size)
    - Random access (no loading everything)
    - Optimal I/O patterns (chunked storage)
    - Works perfectly with DataLoader multiprocessing
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
        hdf5_path: Union[str, Path],
        split: str = 'train',
        patch_size: int = 512,
        augmentation: bool = True,
        cache_in_memory: bool = False
    ):
        """
        Initialize HDF5 dataset.
        
        Args:
            hdf5_path: Path to HDF5 file containing patches
            split: Dataset split ('train', 'val', 'test')
            patch_size: Target patch size for training
            augmentation: Enable data augmentation
            cache_in_memory: Load entire split into RAM (for small datasets)
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.cache_in_memory = cache_in_memory
        
        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {self.hdf5_path}\n"
                f"Create it with: python {__file__} --convert /path/to/dronedeploy/data"
            )
        
        print(f"🗂️  Loading DroneDeploy HDF5 dataset for {split}...")
        print(f"   File: {self.hdf5_path}")
        
        # Validate HDF5 file and get dataset info
        self._validate_and_load_info()
        
        # Setup transforms
        self.transform = self._setup_augmentation()
        
        # Optional: cache entire dataset in memory for maximum speed
        self._memory_cache = None
        if self.cache_in_memory and self.dataset_size < 5000:  # Only for reasonable sizes
            print(f"   🚀 Caching {self.dataset_size} patches in memory...")
            self._load_into_memory()
        
        print(f"   ✅ Ready: {self.dataset_size} patches")
        print(f"   Cache mode: {'Memory' if self._memory_cache else 'HDF5 Random Access'}")
    
    def _validate_and_load_info(self):
        """Validate HDF5 file structure and load dataset info."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Check required groups exist
            required_groups = [f'images/{self.split}', f'masks/{self.split}']
            for group in required_groups:
                if group not in f:
                    raise ValueError(f"Missing required group in HDF5: {group}")
            
            # Get dataset size
            self.dataset_size = f[f'images/{self.split}'].shape[0]
            
            # Load metadata if available
            if 'metadata/dataset_info' in f:
                try:
                    info_str = f['metadata/dataset_info'][()].decode('utf-8')
                    self.dataset_info = json.loads(info_str)
                except:
                    self.dataset_info = {}
            else:
                self.dataset_info = {}
            
            # Get actual patch dimensions
            self.stored_patch_size = f[f'images/{self.split}'].shape[1]
            
            print(f"   HDF5 Info:")
            print(f"     Patches: {self.dataset_size}")
            print(f"     Stored size: {self.stored_patch_size}x{self.stored_patch_size}")
            print(f"     Compression: {f[f'images/{self.split}'].compression}")
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup augmentation pipeline."""
        transforms = []
        
        # Resize if needed
        if self.patch_size != self.stored_patch_size:
            transforms.append(A.Resize(self.patch_size, self.patch_size, always_apply=True))
        
        # Training augmentations
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
        
        # Normalization
        transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        
        return A.Compose(
            transforms,
            additional_targets={'mask': 'mask'}
        )
    
    def _load_into_memory(self):
        """Load entire dataset into memory for maximum speed."""
        self._memory_cache = {'images': [], 'masks': []}
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load all images and masks
            images = f[f'images/{self.split}'][:]
            masks = f[f'masks/{self.split}'][:]
            
            self._memory_cache['images'] = images
            self._memory_cache['masks'] = masks
        
        print(f"   💾 Cached {len(self._memory_cache['images'])} patches in memory")
    
    def __len__(self) -> int:
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a patch by index - ultra-fast HDF5 or memory access."""
        
        if self._memory_cache:
            # Memory cache mode - fastest possible
            image = self._memory_cache['images'][idx].copy()
            mask = self._memory_cache['masks'][idx].copy()
        else:
            # HDF5 random access mode - still very fast
            with h5py.File(self.hdf5_path, 'r') as f:
                image = f[f'images/{self.split}'][idx]
                mask = f[f'masks/{self.split}'][idx]
        
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
            'patch_id': f"{self.split}_{idx:06d}"
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights by sampling the dataset."""
        print("📊 Computing class weights from HDF5...")
        
        class_counts = np.zeros(6)
        sample_size = min(200, len(self))
        
        # Sample evenly across the dataset
        sample_indices = np.linspace(0, len(self) - 1, sample_size, dtype=int)
        
        for i in tqdm(sample_indices, desc="Sampling"):
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
            count = max(class_counts[class_id], 1)
            class_weights[class_id] = total_pixels / (6 * count)
        
        # Apply safety multipliers
        class_weights[3] *= 1.5  # Water
        class_weights[4] *= 2.0  # Car
        
        # Normalize and clip
        class_weights = torch.clamp(class_weights, min=0.5, max=5.0)
        
        return class_weights


def create_hdf5_from_original_tiffs(
    data_root: str,
    output_path: str,
    patch_size: int = 512,
    stride_factor: float = 0.5,
    compression: str = 'gzip',
    compression_level: int = 6
):
    """
    Convert original DroneDeploy TIFF files to HDF5 format.
    
    Args:
        data_root: Path to original DroneDeploy data (with images/ and labels/ dirs)
        output_path: Path for output HDF5 file
        patch_size: Size of patches to extract
        stride_factor: Overlap factor for patch extraction
        compression: HDF5 compression algorithm
        compression_level: Compression level (0-9)
    """
    data_root = Path(data_root)
    images_dir = data_root / 'images'
    labels_dir = data_root / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Invalid data structure. Need {images_dir} and {labels_dir}")
    
    print(f"🔄 Converting DroneDeploy TIFFs to HDF5...")
    print(f"   Source: {data_root}")
    print(f"   Output: {output_path}")
    print(f"   Patch size: {patch_size}x{patch_size}")
    print(f"   Compression: {compression} level {compression_level}")
    
    # Create class mapping lookup table
    mapping_lut = np.full(256, 5, dtype=np.uint8)
    for dd_class, landing_class in DroneDeployHDF5Dataset.DRONEDEPLOY_TO_LANDING.items():
        if dd_class < 256:
            mapping_lut[dd_class] = landing_class
    
    # Find matching image/label pairs
    image_files = []
    for img_path in images_dir.glob("*.tif"):
        img_stem = img_path.stem
        if img_stem.endswith('-ortho'):
            base_name = img_stem[:-6]
            label_name = f"{base_name}-label.png"
        else:
            label_name = f"{img_stem}-label.png"
        
        label_path = labels_dir / label_name
        if label_path.exists():
            image_files.append((img_path, label_path))
    
    print(f"   Found {len(image_files)} image pairs")
    
    # Extract all patches first
    all_patches = []
    stride = int(patch_size * stride_factor)
    
    for img_path, label_path in tqdm(image_files, desc="Extracting patches"):
        try:
            # Load images
            image = cv2.imread(str(img_path))
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or label is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Extract patches
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    img_patch = image[y:y+patch_size, x:x+patch_size]
                    label_patch = label[y:y+patch_size, x:x+patch_size]
                    
                    # Map to landing classes
                    landing_patch = mapping_lut[np.clip(label_patch, 0, 255)]
                    
                    # Quality check
                    valid_pixels = np.sum(landing_patch != 5)
                    if valid_pixels / landing_patch.size > 0.1:  # At least 10% valid
                        unique_classes = len(np.unique(landing_patch))
                        if unique_classes >= 2:  # Multiple classes
                            all_patches.append({
                                'image': img_patch,
                                'mask': landing_patch,
                                'source': img_path.stem
                            })
        except Exception as e:
            print(f"   ⚠️  Error processing {img_path.name}: {e}")
    
    print(f"   Extracted {len(all_patches)} valid patches")
    
    # Create deterministic splits
    np.random.seed(42)  # Reproducible splits
    indices = np.random.permutation(len(all_patches))
    
    n_train = int(0.6 * len(all_patches))
    n_val = int(0.8 * len(all_patches))
    
    splits = {
        'train': indices[:n_train],
        'val': indices[n_train:n_val],
        'test': indices[n_val:]
    }
    
    # Create HDF5 file
    print(f"   Creating HDF5 file...")
    
    with h5py.File(output_path, 'w') as f:
        # Create groups
        images_group = f.create_group('images')
        masks_group = f.create_group('masks')
        metadata_group = f.create_group('metadata')
        
        for split_name, split_indices in splits.items():
            split_patches = [all_patches[i] for i in split_indices]
            n_patches = len(split_patches)
            
            if n_patches == 0:
                continue
            
            print(f"   Writing {split_name}: {n_patches} patches")
            
            # Create datasets with optimal chunking and compression
            chunk_size = min(64, n_patches)  # Optimal chunk size
            
            images_ds = images_group.create_dataset(
                split_name,
                shape=(n_patches, patch_size, patch_size, 3),
                dtype=np.uint8,
                compression=compression,
                compression_opts=compression_level,
                chunks=(chunk_size, patch_size, patch_size, 3),
                shuffle=True,  # Improve compression
                fletcher32=True  # Checksums
            )
            
            masks_ds = masks_group.create_dataset(
                split_name,
                shape=(n_patches, patch_size, patch_size),
                dtype=np.uint8,
                compression=compression,
                compression_opts=compression_level,
                chunks=(chunk_size, patch_size, patch_size),
                shuffle=True,
                fletcher32=True
            )
            
            # Write data in batches
            batch_size = 100
            for i in range(0, n_patches, batch_size):
                end_i = min(i + batch_size, n_patches)
                batch_patches = split_patches[i:end_i]
                
                batch_images = np.stack([p['image'] for p in batch_patches])
                batch_masks = np.stack([p['mask'] for p in batch_patches])
                
                images_ds[i:end_i] = batch_images
                masks_ds[i:end_i] = batch_masks
        
        # Store metadata
        dataset_info = {
            'created_from': str(data_root),
            'patch_size': patch_size,
            'stride_factor': stride_factor,
            'total_patches': len(all_patches),
            'splits': {k: len(v) for k, v in splits.items()},
            'class_mapping': DroneDeployHDF5Dataset.DRONEDEPLOY_TO_LANDING,
            'compression': compression,
            'compression_level': compression_level
        }
        
        metadata_group.create_dataset(
            'dataset_info',
            data=json.dumps(dataset_info).encode('utf-8')
        )
    
    print(f"✅ HDF5 conversion complete!")
    print(f"   Output: {output_path}")
    print(f"   Splits: {dict((k, len(v)) for k, v in splits.items())}")
    
    # Report file size
    file_size = Path(output_path).stat().st_size / 1024**3
    print(f"   File size: {file_size:.2f} GB")


def create_hdf5_datasets(
    hdf5_path: str,
    patch_size: int = 512,
    **kwargs
) -> Dict[str, DroneDeployHDF5Dataset]:
    """
    Create HDF5-based DroneDeploy datasets.
    
    Args:
        hdf5_path: Path to HDF5 file
        patch_size: Target patch size
        **kwargs: Additional dataset parameters
        
    Returns:
        Dictionary with train/val/test datasets
    """
    datasets = {}
    
    # Check which splits are available
    with h5py.File(hdf5_path, 'r') as f:
        available_splits = []
        for split in ['train', 'val', 'test']:
            if f'images/{split}' in f:
                available_splits.append(split)
    
    print(f"📊 Available splits in HDF5: {available_splits}")
    
    for split in available_splits:
        try:
            dataset = DroneDeployHDF5Dataset(
                hdf5_path=hdf5_path,
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
    parser = argparse.ArgumentParser(description='DroneDeploy HDF5 Dataset Tools')
    
    parser.add_argument('--convert', type=str, metavar='DATA_ROOT',
                        help='Convert DroneDeploy TIFFs to HDF5 format')
    parser.add_argument('--output', type=str, default='dronedeploy_patches.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Patch size for extraction')
    parser.add_argument('--test', type=str, metavar='HDF5_FILE',
                        help='Test loading from HDF5 file')
    
    args = parser.parse_args()
    
    if args.convert:
        # Convert TIFFs to HDF5
        print("🔄 Converting DroneDeploy data to HDF5...")
        create_hdf5_from_original_tiffs(
            data_root=args.convert,
            output_path=args.output,
            patch_size=args.patch_size
        )
        print(f"✅ Conversion complete: {args.output}")
        
    elif args.test:
        # Test HDF5 dataset loading
        print(f"🧪 Testing HDF5 dataset: {args.test}")
        
        try:
            datasets = create_hdf5_datasets(
                hdf5_path=args.test,
                patch_size=512,
                augmentation=True
            )
            
            print(f"\n✅ Dataset loading successful!")
            for split, dataset in datasets.items():
                if dataset is not None:
                    print(f"   {split}: {len(dataset)} patches")
            
            # Test sample loading
            if datasets.get('train') and len(datasets['train']) > 0:
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
    
    else:
        print("DroneDeploy HDF5 Dataset Tools")
        print("=" * 40)
        print("Convert TIFFs to HDF5:")
        print("  python dronedeploy_hdf5_dataset.py --convert /path/to/dronedeploy")
        print()
        print("Test HDF5 loading:")
        print("  python dronedeploy_hdf5_dataset.py --test dronedeploy_patches.h5")
        print()
        print("Use in training:")
        print("  datasets = create_hdf5_datasets('dronedeploy_patches.h5')") 