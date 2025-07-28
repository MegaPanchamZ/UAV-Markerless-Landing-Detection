#!/usr/bin/env python3
"""
Checkpoint Inspector - Analyze saved model structure
"""

import torch
from collections import OrderedDict

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint structure to understand the saved architecture."""
    
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n📊 Checkpoint Keys:")
    for key in checkpoint.keys():
        print(f"   {key}")
    
    print(f"\n🏗️  Model State Dict Structure:")
    state_dict = checkpoint['model_state_dict']
    
    # Group keys by component
    backbone_keys = []
    seg_head_keys = []
    uncertainty_keys = []
    other_keys = []
    
    for key in sorted(state_dict.keys()):
        shape = tuple(state_dict[key].shape)
        print(f"   {key:<50} {shape}")
        
        if key.startswith('features.'):
            backbone_keys.append((key, shape))
        elif key.startswith('seg_head.'):
            seg_head_keys.append((key, shape))
        elif key.startswith('uncertainty_head.'):
            uncertainty_keys.append((key, shape))
        else:
            other_keys.append((key, shape))
    
    print(f"\n📋 Component Analysis:")
    print(f"   Backbone (features.*): {len(backbone_keys)} layers")
    print(f"   Segmentation Head: {len(seg_head_keys)} layers")
    print(f"   Uncertainty Head: {len(uncertainty_keys)} layers")
    print(f"   Other: {len(other_keys)} layers")
    
    print(f"\n🔧 Segmentation Head Architecture:")
    for key, shape in seg_head_keys:
        print(f"   {key:<30} {shape}")
    
    print(f"\n🎯 Uncertainty Head Architecture:")
    for key, shape in uncertainty_keys:
        print(f"   {key:<30} {shape}")
    
    # Additional metadata
    if 'epoch' in checkpoint:
        print(f"\n📈 Training Info:")
        print(f"   Epoch: {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        print(f"   Metrics: {checkpoint['metrics']}")

if __name__ == "__main__":
    inspect_checkpoint("uav_landing_project/outputs/stage2_best.pth") 