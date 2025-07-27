#!/usr/bin/env python3
"""
DroneDeploy Dataset Preprocessor
===============================

Comprehensive preprocessing script to convert large DroneDeploy GeoTIFF files
into optimized formats for ultra-fast training.

Supported output formats:
1. Individual files: Traditional approach with separate image files
2. HDF5 format: Single optimized file with compression (RECOMMENDED)
3. Optimized directory: Simple images/ and masks/ structure

This solves the architectural bottleneck by:
1. Reading each large TIFF once 
2. Extracting all valid patches
3. Converting to optimal format for training
4. Creating proper train/val/test splits

Usage:
    # Convert to HDF5 (recommended for best performance)
    python dronedeploy_preprocessor.py --source /path/to/dronedeploy --output-hdf5 dronedeploy_patches.h5
    
    # Convert to optimized directory structure
    python dronedeploy_preprocessor.py --source /path/to/dronedeploy --output-dir /path/to/optimized
    
    # Convert to individual files (legacy)
    python dronedeploy_preprocessor.py --source /path/to/dronedeploy --output-legacy /path/to/legacy
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import hashlib
import json
import argparse

# Optional HDF5 support
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("⚠️  HDF5 not available. Install with: pip install h5py")

class DroneDeployPreprocessor:
    """
    Comprehensive DroneDeploy dataset preprocessor with multiple output formats.
    """
    
    # DroneDeploy classes → Landing classes (same as original)
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
        source_data_root: str,
        patch_size: int = 512,
        stride_factor: float = 0.5,
        min_valid_pixels: float = 0.1,
        quality_threshold: int = 2  # Minimum number of unique classes
    ):
        self.source_root = Path(source_data_root)
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_factor)
        self.min_valid_pixels = min_valid_pixels
        self.quality_threshold = quality_threshold
        
        # Validate source structure
        if not (self.source_root / 'images').exists():
            raise ValueError(f"Source directory must contain 'images' folder: {self.source_root}")
        if not (self.source_root / 'labels').exists():
            raise ValueError(f"Source directory must contain 'labels' folder: {self.source_root}")
        
        # Create fast lookup table for class mapping
        self.mapping_lut = np.full(256, 5, dtype=np.uint8)  # Default to clutter
        for dd_class, landing_class in self.DRONEDEPLOY_TO_LANDING.items():
            if dd_class < 256:
                self.mapping_lut[dd_class] = landing_class
        
        print(f"🔧 DroneDeploy Preprocessor initialized:")
        print(f"   Source: {self.source_root}")
        print(f"   Patch size: {patch_size}x{patch_size}")
        print(f"   Stride: {self.stride} ({stride_factor * 100}% overlap)")
        print(f"   Quality filters: {min_valid_pixels * 100}% valid pixels, {quality_threshold}+ classes")
    
    def find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching image and label file pairs."""
        images_dir = self.source_root / 'images'
        labels_dir = self.source_root / 'labels'
        
        image_files = []
        
        print(f"🔍 Scanning for image pairs...")
        
        # Find all .tif images
        for img_path in images_dir.glob("*.tif"):
            # Handle naming pattern: image_name-ortho.tif → image_name-label.png
            img_stem = img_path.stem
            
            if img_stem.endswith('-ortho'):
                base_name = img_stem[:-6]  # Remove '-ortho'
                label_name = f"{base_name}-label.png"
            else:
                label_name = f"{img_stem}-label.png"
            
            label_path = labels_dir / label_name
            
            if label_path.exists():
                # Check file sizes
                try:
                    img_size = img_path.stat().st_size
                    label_size = label_path.stat().st_size
                    
                    if img_size > 1024 * 1024 and label_size > 1024:
                        image_files.append((img_path, label_path))
                        print(f"   ✓ {img_path.name} → {label_name}")
                    else:
                        print(f"   ⚠️  Skipping small file: {img_path.name}")
                        
                except Exception as e:
                    print(f"   ❌ Error checking {img_path.name}: {e}")
            else:
                print(f"   ⚠️  No label found for: {img_path.name}")
        
        print(f"✅ Found {len(image_files)} valid image pairs")
        return sorted(image_files)
    
    def map_to_landing_classes(self, dronedeploy_label: np.ndarray) -> np.ndarray:
        """Map DroneDeploy classes to landing classes using fast vectorized lookup."""
        clipped_label = np.clip(dronedeploy_label, 0, 255)
        return self.mapping_lut[clipped_label]
    
    def is_valid_patch(self, label_patch: np.ndarray) -> bool:
        """Check if patch has sufficient valid content."""
        # Count non-clutter pixels
        valid_pixels = np.sum(label_patch != 5)  # Not clutter
        total_pixels = label_patch.size
        valid_ratio = valid_pixels / total_pixels
        
        # Require minimum valid content
        if valid_ratio < self.min_valid_pixels:
            return False
        
        # Prefer patches with multiple classes
        unique_classes = len(np.unique(label_patch))
        if unique_classes < self.quality_threshold:
            return False
        
        # Bonus: prioritize patches with safety-critical classes
        has_critical = np.any([
            np.any(label_patch == 3),  # Water (critical hazard)
            np.any(label_patch == 4),  # Car (dynamic obstacle)
            np.any(label_patch == 2)   # Building (obstacle)
        ])
        
        return True
    
    def extract_all_patches(self) -> List[Dict]:
        """Extract all valid patches from all image pairs."""
        
        image_pairs = self.find_image_pairs()
        
        if not image_pairs:
            raise ValueError("No valid image pairs found!")
        
        print(f"\n🔄 Extracting patches from {len(image_pairs)} images...")
        
        all_patches = []
        
        for i, (img_path, label_path) in enumerate(tqdm(image_pairs, desc="Processing images")):
            try:
                # Load images
                image = cv2.imread(str(img_path))
                label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                
                if image is None or label is None:
                    print(f"   ❌ Failed to load: {img_path.name}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                
                # Extract patches from this image
                image_patches = 0
                for y in range(0, h - self.patch_size + 1, self.stride):
                    for x in range(0, w - self.patch_size + 1, self.stride):
                        # Extract patch
                        img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                        label_patch = label[y:y+self.patch_size, x:x+self.patch_size]
                        
                        # Map to landing classes
                        landing_patch = self.map_to_landing_classes(label_patch)
                        
                        # Validate patch
                        if self.is_valid_patch(landing_patch):
                            # Store patch data
                            patch_id = f"img_{i:03d}_{x:04d}_{y:04d}"
                            
                            all_patches.append({
                                'patch_id': patch_id,
                                'image_patch': img_patch,
                                'mask_patch': landing_patch,
                                'source_image': img_path.name,
                                'source_label': label_path.name,
                                'coordinates': (x, y),
                                'image_index': i
                            })
                            
                            image_patches += 1
                
                print(f"   📷 {img_path.name}: {image_patches} patches")
                
            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")
                continue
        
        print(f"\n✅ Extracted {len(all_patches)} total patches")
        return all_patches
    
    def create_splits(self, all_patches: List[Dict]) -> Dict[str, List[int]]:
        """Create deterministic train/val/test splits by source image."""
        
        # Group patches by source image
        patches_by_source = {}
        for i, patch in enumerate(all_patches):
            source = patch['source_image']
            if source not in patches_by_source:
                patches_by_source[source] = []
            patches_by_source[source].append(i)
        
        # Sort source images deterministically
        source_images = sorted(patches_by_source.keys())
        
        # Create deterministic split based on hash (6:2:2)
        train_indices, val_indices, test_indices = [], [], []
        
        for source in source_images:
            # Use hash for deterministic assignment
            hash_val = int(hashlib.md5(source.encode()).hexdigest(), 16)
            split_val = hash_val % 10
            
            indices = patches_by_source[source]
            
            if split_val < 6:  # 60% train
                train_indices.extend(indices)
            elif split_val < 8:  # 20% val
                val_indices.extend(indices)
            else:  # 20% test
                test_indices.extend(indices)
        
        print(f"\n📊 Dataset splits:")
        print(f"   Train: {len(train_indices)} patches ({len(train_indices)/len(all_patches)*100:.1f}%)")
        print(f"   Val: {len(val_indices)} patches ({len(val_indices)/len(all_patches)*100:.1f}%)")
        print(f"   Test: {len(test_indices)} patches ({len(test_indices)/len(all_patches)*100:.1f}%)")
        
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
    
    def save_as_hdf5(self, all_patches: List[Dict], output_path: str, compression: str = 'gzip'):
        """Save patches to HDF5 format (RECOMMENDED)."""
        
        if not HDF5_AVAILABLE:
            raise ImportError("HDF5 not available. Install with: pip install h5py")
        
        print(f"\n🗂️  Saving to HDF5 format: {output_path}")
        print(f"   Compression: {compression}")
        
        # Create splits
        splits = self.create_splits(all_patches)
        
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
                
                # Optimal chunking and compression
                chunk_size = min(64, n_patches)
                
                # Create datasets
                images_ds = images_group.create_dataset(
                    split_name,
                    shape=(n_patches, self.patch_size, self.patch_size, 3),
                    dtype=np.uint8,
                    compression=compression,
                    compression_opts=6,
                    chunks=(chunk_size, self.patch_size, self.patch_size, 3),
                    shuffle=True,  # Improve compression
                    fletcher32=True  # Checksums
                )
                
                masks_ds = masks_group.create_dataset(
                    split_name,
                    shape=(n_patches, self.patch_size, self.patch_size),
                    dtype=np.uint8,
                    compression=compression,
                    compression_opts=6,
                    chunks=(chunk_size, self.patch_size, self.patch_size),
                    shuffle=True,
                    fletcher32=True
                )
                
                # Write data in batches
                batch_size = 100
                for i in range(0, n_patches, batch_size):
                    end_i = min(i + batch_size, n_patches)
                    batch_patches = split_patches[i:end_i]
                    
                    batch_images = np.stack([p['image_patch'] for p in batch_patches])
                    batch_masks = np.stack([p['mask_patch'] for p in batch_patches])
                    
                    images_ds[i:end_i] = batch_images
                    masks_ds[i:end_i] = batch_masks
            
            # Store metadata
            dataset_info = {
                'created_from': str(self.source_root),
                'patch_size': self.patch_size,
                'stride': self.stride,
                'stride_factor': self.stride / self.patch_size,
                'min_valid_pixels': self.min_valid_pixels,
                'quality_threshold': self.quality_threshold,
                'total_patches': len(all_patches),
                'splits': {k: len(v) for k, v in splits.items()},
                'class_mapping': self.DRONEDEPLOY_TO_LANDING,
                'landing_classes': self.LANDING_CLASSES,
                'compression': compression
            }
            
            metadata_group.create_dataset(
                'dataset_info',
                data=json.dumps(dataset_info, indent=2).encode('utf-8')
            )
        
        # Report results
        file_size = Path(output_path).stat().st_size / 1024**3
        print(f"✅ HDF5 creation complete!")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size:.2f} GB")
        print(f"   Compression ratio: ~{len(all_patches) * 0.5 / 1024 / file_size:.1f}x")
    
    def save_as_optimized_directory(self, all_patches: List[Dict], output_dir: str):
        """Save patches to optimized directory structure (images/ + masks/ + splits/)."""
        
        output_path = Path(output_dir)
        images_dir = output_path / 'images'
        masks_dir = output_path / 'masks'
        splits_dir = output_path / 'splits'
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Saving to optimized directory: {output_path}")
        
        # Create splits
        splits = self.create_splits(all_patches)
        
        # Save all patches as individual files
        print(f"   Writing {len(all_patches)} patch files...")
        for patch in tqdm(all_patches, desc="Saving patches"):
            patch_id = patch['patch_id']
            
            # Save image
            img_file = images_dir / f"{patch_id}.jpg"
            cv2.imwrite(str(img_file), cv2.cvtColor(patch['image_patch'], cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save mask
            mask_file = masks_dir / f"{patch_id}.png"
            cv2.imwrite(str(mask_file), patch['mask_patch'])
        
        # Save split files
        for split_name, split_indices in splits.items():
            split_file = splits_dir / f"{split_name}.txt"
            split_patch_ids = [all_patches[i]['patch_id'] for i in split_indices]
            
            with open(split_file, 'w') as f:
                for patch_id in split_patch_ids:
                    f.write(f"{patch_id}\n")
            
            print(f"   📝 {split_name}: {len(split_patch_ids)} patches → {split_file}")
        
        # Save metadata
        metadata = {
            'created_from': str(self.source_root),
            'patch_size': self.patch_size,
            'stride': self.stride,
            'stride_factor': self.stride / self.patch_size,
            'min_valid_pixels': self.min_valid_pixels,
            'quality_threshold': self.quality_threshold,
            'total_patches': len(all_patches),
            'splits': {k: len(v) for k, v in splits.items()},
            'class_mapping': self.DRONEDEPLOY_TO_LANDING,
            'landing_classes': self.LANDING_CLASSES,
            'structure': {
                'images': 'images/*.jpg',
                'masks': 'masks/*.png',
                'splits': 'splits/train.txt, val.txt, test.txt'
            }
        }
        
        metadata_file = output_path / 'dataset_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Optimized directory creation complete!")
        print(f"   Structure: {output_path}")
        print(f"   Usage: DroneDeployOptimizedDataset('{output_path}')")
    
    def save_as_legacy_files(self, all_patches: List[Dict], output_dir: str):
        """Save patches to legacy individual files format with CSV metadata."""
        
        output_path = Path(output_dir)
        patches_dir = output_path / "patches"
        images_dir = patches_dir / "images"
        labels_dir = patches_dir / "labels"
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 Saving to legacy format: {output_path}")
        
        # Create splits
        splits = self.create_splits(all_patches)
        
        # Save patches and create metadata
        patch_metadata = []
        
        print(f"   Writing {len(all_patches)} patch files...")
        for patch in tqdm(all_patches, desc="Saving patches"):
            patch_id = patch['patch_id']
            
            # Save image
            img_file = images_dir / f"{patch_id}.png"
            cv2.imwrite(str(img_file), cv2.cvtColor(patch['image_patch'], cv2.COLOR_RGB2BGR))
            
            # Save mask
            label_file = labels_dir / f"{patch_id}.png"
            cv2.imwrite(str(label_file), patch['mask_patch'])
            
            # Store metadata
            patch_metadata.append({
                'patch_id': patch_id,
                'image_file': f"patches/images/{patch_id}.png",
                'label_file': f"patches/labels/{patch_id}.png",
                'source_image': patch['source_image'],
                'source_label': patch['source_label'],
                'x': patch['coordinates'][0],
                'y': patch['coordinates'][1],
                'patch_size': self.patch_size
            })
        
        # Save split metadata files
        for split_name, split_indices in splits.items():
            split_metadata = [patch_metadata[i] for i in split_indices]
            metadata_file = output_path / f"{split_name}_metadata.csv"
            
            df = pd.DataFrame(split_metadata)
            df.to_csv(metadata_file, index=False)
            print(f"   📝 {split_name}: {len(split_metadata)} patches → {metadata_file}")
        
        # Save processing parameters
        params = {
            'patch_size': self.patch_size,
            'stride': self.stride,
            'stride_factor': self.stride / self.patch_size,
            'min_valid_pixels': self.min_valid_pixels,
            'quality_threshold': self.quality_threshold,
            'total_patches': len(all_patches),
            'source_images': len(set(p['source_image'] for p in all_patches)),
            'class_mapping': self.DRONEDEPLOY_TO_LANDING,
            'landing_classes': self.LANDING_CLASSES
        }
        
        params_file = output_path / "preprocessing_params.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✅ Legacy format creation complete!")
        print(f"   Structure: {output_path}")
        print(f"   Usage: OptimizedDroneDeployDataset('{output_path}')")


def main():
    """Main preprocessing script with multiple output format options."""
    
    parser = argparse.ArgumentParser(
        description='Preprocess DroneDeploy dataset for fast training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Format Options (choose one):

  --output-hdf5 FILE       Save to HDF5 format (RECOMMENDED for best performance)
                          Single compressed file with optimal access patterns
                          
  --output-dir DIR         Save to optimized directory structure
                          Simple images/ + masks/ + splits/ layout
                          
  --output-legacy DIR      Save to legacy individual files format
                          Compatible with existing OptimizedDroneDeployDataset

Examples:
  # Convert to HDF5 (best performance)
  python dronedeploy_preprocessor.py --source /data/dronedeploy --output-hdf5 patches.h5
  
  # Convert to optimized directory
  python dronedeploy_preprocessor.py --source /data/dronedeploy --output-dir /data/optimized
        """
    )
    
    # Required arguments
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source DroneDeploy dataset (with images/ and labels/ dirs)')
    
    # Output format options (mutually exclusive)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--output-hdf5', type=str, metavar='FILE',
                              help='Output HDF5 file (RECOMMENDED)')
    output_group.add_argument('--output-dir', type=str, metavar='DIR',
                              help='Output optimized directory')
    output_group.add_argument('--output-legacy', type=str, metavar='DIR',
                              help='Output legacy individual files')
    
    # Processing parameters
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Patch size (default: 512)')
    parser.add_argument('--stride-factor', type=float, default=0.5,
                        help='Stride factor (default: 0.5 = 50%% overlap)')
    parser.add_argument('--min-valid-pixels', type=float, default=0.1,
                        help='Minimum valid pixel ratio (default: 0.1)')
    parser.add_argument('--quality-threshold', type=int, default=2,
                        help='Minimum unique classes per patch (default: 2)')
    parser.add_argument('--compression', type=str, default='gzip', 
                        choices=['gzip', 'lzf', 'szip'],
                        help='HDF5 compression algorithm (default: gzip)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.output_hdf5 and not HDF5_AVAILABLE:
        print("❌ HDF5 output requested but h5py not available.")
        print("   Install with: pip install h5py")
        return
    
    # Create preprocessor
    print("🚀 DroneDeploy Dataset Preprocessor")
    print("=" * 40)
    
    preprocessor = DroneDeployPreprocessor(
        source_data_root=args.source,
        patch_size=args.patch_size,
        stride_factor=args.stride_factor,
        min_valid_pixels=args.min_valid_pixels,
        quality_threshold=args.quality_threshold
    )
    
    # Extract all patches once
    all_patches = preprocessor.extract_all_patches()
    
    if not all_patches:
        print("❌ No valid patches extracted!")
        return
    
    # Save in requested format
    if args.output_hdf5:
        preprocessor.save_as_hdf5(all_patches, args.output_hdf5, args.compression)
        
        print(f"\n🎉 HDF5 preprocessing complete!")
        print(f"   Use with: python train.py --stage 2 --use-hdf5 {args.output_hdf5}")
        
    elif args.output_dir:
        preprocessor.save_as_optimized_directory(all_patches, args.output_dir)
        
        print(f"\n🎉 Optimized directory preprocessing complete!")
        print(f"   Use with: DroneDeployOptimizedDataset('{args.output_dir}')")
        
    elif args.output_legacy:
        preprocessor.save_as_legacy_files(all_patches, args.output_legacy)
        
        print(f"\n🎉 Legacy preprocessing complete!")
        print(f"   Use with: OptimizedDroneDeployDataset('{args.output_legacy}')")
    
    # Performance recommendations
    print(f"\n💡 Performance Recommendations:")
    if args.output_hdf5:
        print(f"   • HDF5 format provides the best training performance")
        print(f"   • Use --hdf5-cache-memory for even faster loading")
        print(f"   • Single file is easy to deploy and backup")
    else:
        print(f"   • Consider using HDF5 format for optimal performance:")
        print(f"     python {__file__} --source {args.source} --output-hdf5 patches.h5")


if __name__ == "__main__":
    main() 