#!/usr/bin/env python3
"""
Progressive Class Mapping System for Multi-Stage Training
=========================================================

Manages class mapping transitions between different training stages:
Stage 1: Semantic Drone Dataset (24 native classes)
Stage 2: DroneDeploy Dataset (6 classes) 
Stage 3: UDD6 Dataset (6 classes)

Provides utilities for:
- Class mapping between stages
- Knowledge transfer between models with different output dimensions
- Consistent semantic interpretation across datasets
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class ProgressiveClassMapper:
    """
    Manages class mapping transitions for progressive training.
    """
    
    def __init__(self):
        """Initialize progressive class mapping system."""
        
        # Stage 1: Native 24 classes from Semantic Drone Dataset
        self.stage1_classes = {
            0: "unlabeled", 1: "paved-area", 2: "dirt", 3: "grass", 4: "gravel",
            5: "water", 6: "rocks", 7: "pool", 8: "vegetation", 9: "roof",
            10: "wall", 11: "window", 12: "door", 13: "fence", 14: "fence-pole",
            15: "person", 16: "dog", 17: "car", 18: "bicycle", 19: "tree",
            20: "bald-tree", 21: "ar-marker", 22: "obstacle", 23: "conflicting"
        }
        
        # Stage 2: DroneDeploy 6 classes
        self.stage2_classes = {
            0: "ground",     # Safe flat surfaces
            1: "vegetation", # Trees, grass
            2: "building",   # Buildings, structures  
            3: "water",      # Water bodies
            4: "car",        # Vehicles, moving objects
            5: "clutter"     # Unknown, mixed areas
        }
        
        # Stage 3: UDD6 6 classes (urban-focused, no water)
        self.stage3_classes = {
            0: "other",      # Background/unknown
            1: "facade",     # Building facades 
            2: "road",       # Roads/ground surfaces
            3: "vegetation", # Vegetation 
            4: "vehicle",    # Vehicles
            5: "roof"        # Roofs
        }
        
        # Define semantic mapping relationships
        self._setup_progressive_mappings()
    
    def _setup_progressive_mappings(self):
        """Setup mapping relationships between stages."""
        
        # Stage 1 → Stage 2 mapping (24 classes → 6 classes)
        # Based on semantic similarity and landing safety assessment
        self.stage1_to_stage2 = {
            # Ground: Safe, flat landing surfaces
            1: 0,  # paved-area → ground
            2: 0,  # dirt → ground  
            4: 0,  # gravel → ground
            
            # Vegetation: Natural surfaces, emergency landing possible
            3: 1,  # grass → vegetation
            8: 1,  # vegetation → vegetation
            19: 1, # tree → vegetation
            20: 1, # bald-tree → vegetation
            
            # Building: Structures, obstacles to avoid
            9: 2,  # roof → building
            10: 2, # wall → building
            11: 2, # window → building
            12: 2, # door → building
            13: 2, # fence → building
            14: 2, # fence-pole → building
            
            # Water: Critical hazard for landing
            5: 3,  # water → water
            7: 3,  # pool → water
            
            # Car: Dynamic obstacles, moving objects
            17: 4, # car → car
            18: 4, # bicycle → car
            15: 4, # person → car (moving obstacle)
            16: 4, # dog → car (moving obstacle)
            
            # Clutter: Unknown, ambiguous, or complex areas
            0: 5,  # unlabeled → clutter
            6: 5,  # rocks → clutter
            21: 5, # ar-marker → clutter
            22: 5, # obstacle → clutter
            23: 5  # conflicting → clutter
        }
        
        # Stage 2 → Stage 3 mapping (6 DroneDeploy → 6 UDD6)
        # Handle the fact that UDD6 is urban-focused and has no water class
        self.stage2_to_stage3 = {
            0: 2,  # ground → road (urban ground surfaces)
            1: 3,  # vegetation → vegetation (direct match)
            2: 1,  # building → facade (building structures in urban context)
            3: 0,  # water → other (no water in urban UDD6, map to background)
            4: 4,  # car → vehicle (direct match)
            5: 0   # clutter → other (unknown areas to background)
        }
        
        # UDD6 native mapping (for direct Stage 3 training)
        self.udd6_native_to_stage3 = {
            0: 0,  # other → other
            1: 1,  # facade → facade
            2: 2,  # road → road
            3: 3,  # vegetation → vegetation
            4: 4,  # vehicle → vehicle
            5: 5   # roof → roof
        }
    
    def get_mapping_matrix(self, source_stage: int, target_stage: int) -> torch.Tensor:
        """
        Create a mapping matrix for class conversion between stages.
        
        Args:
            source_stage: Source stage number (1, 2, or 3)
            target_stage: Target stage number (1, 2, or 3)
            
        Returns:
            Mapping matrix [target_classes, source_classes] for probability conversion
        """
        
        if source_stage == 1 and target_stage == 2:
            # 24 classes → 6 classes
            mapping = self.stage1_to_stage2
            source_classes = 24
            target_classes = 6
        elif source_stage == 2 and target_stage == 3:
            # 6 classes → 6 classes
            mapping = self.stage2_to_stage3
            source_classes = 6
            target_classes = 6
        else:
            raise ValueError(f"Unsupported mapping: Stage {source_stage} → Stage {target_stage}")
        
        # Create mapping matrix
        matrix = torch.zeros(target_classes, source_classes)
        
        for source_class, target_class in mapping.items():
            matrix[target_class, source_class] = 1.0
        
        # Normalize to handle many-to-one mappings
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = matrix / (row_sums + 1e-8)  # Avoid division by zero
        
        return matrix
    
    def convert_predictions(
        self, 
        predictions: torch.Tensor, 
        source_stage: int, 
        target_stage: int
    ) -> torch.Tensor:
        """
        Convert model predictions between different stage class schemes.
        
        Args:
            predictions: Model predictions [B, C, H, W] (logits or probabilities)
            source_stage: Source stage number
            target_stage: Target stage number
            
        Returns:
            Converted predictions in target stage class scheme
        """
        
        device = predictions.device
        mapping_matrix = self.get_mapping_matrix(source_stage, target_stage).to(device)
        
        # Convert logits to probabilities if needed
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.softmax(predictions, dim=1)
        
        # Apply mapping matrix: [B, C_target, H, W]
        B, C, H, W = predictions.shape
        predictions_flat = predictions.view(B, C, -1)  # [B, C_source, H*W]
        
        # Matrix multiplication: [C_target, C_source] × [B, C_source, H*W] → [B, C_target, H*W]
        mapped_predictions = torch.bmm(
            mapping_matrix.unsqueeze(0).expand(B, -1, -1),
            predictions_flat
        )
        
        # Reshape back
        mapped_predictions = mapped_predictions.view(B, -1, H, W)
        
        return mapped_predictions
    
    def convert_labels(
        self, 
        labels: torch.Tensor, 
        source_stage: int, 
        target_stage: int
    ) -> torch.Tensor:
        """
        Convert ground truth labels between different stage class schemes.
        
        Args:
            labels: Ground truth labels [B, H, W] (class indices)
            source_stage: Source stage number
            target_stage: Target stage number
            
        Returns:
            Converted labels in target stage class scheme
        """
        
        if source_stage == 1 and target_stage == 2:
            mapping = self.stage1_to_stage2
        elif source_stage == 2 and target_stage == 3:
            mapping = self.stage2_to_stage3
        else:
            raise ValueError(f"Unsupported mapping: Stage {source_stage} → Stage {target_stage}")
        
        # Create lookup table
        max_class = max(mapping.keys()) + 1
        lookup = torch.zeros(max_class, dtype=torch.long, device=labels.device)
        
        for source_class, target_class in mapping.items():
            lookup[source_class] = target_class
        
        # Apply mapping
        converted_labels = lookup[labels.clamp(0, max_class - 1)]
        
        return converted_labels
    
    def get_class_weights_transfer(
        self, 
        source_weights: torch.Tensor, 
        source_stage: int, 
        target_stage: int
    ) -> torch.Tensor:
        """
        Transfer class weights between stages based on class mapping.
        
        Args:
            source_weights: Class weights from source stage [C_source]
            source_stage: Source stage number
            target_stage: Target stage number
            
        Returns:
            Transferred class weights for target stage [C_target]
        """
        
        mapping_matrix = self.get_mapping_matrix(source_stage, target_stage)
        
        # Weight transfer: average weights of mapped classes
        target_weights = torch.mv(mapping_matrix, source_weights)
        
        return target_weights
    
    def save_mapping_config(self, save_path: str):
        """Save mapping configuration to file."""
        config = {
            'stage1_classes': self.stage1_classes,
            'stage2_classes': self.stage2_classes,
            'stage3_classes': self.stage3_classes,
            'stage1_to_stage2': self.stage1_to_stage2,
            'stage2_to_stage3': self.stage2_to_stage3,
            'udd6_native_to_stage3': self.udd6_native_to_stage3
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Progressive mapping config saved to: {save_path}")
    
    def print_mapping_summary(self):
        """Print a summary of all class mappings."""
        print("🔄 Progressive Class Mapping Summary")
        print("=" * 50)
        
        print("\n📊 Stage 1: Semantic Drone Dataset (24 classes)")
        for class_id, name in self.stage1_classes.items():
            print(f"   {class_id:2d}: {name}")
        
        print(f"\n📊 Stage 2: DroneDeploy Dataset (6 classes)")
        for class_id, name in self.stage2_classes.items():
            print(f"   {class_id}: {name}")
        
        print(f"\n📊 Stage 3: UDD6 Dataset (6 classes)")
        for class_id, name in self.stage3_classes.items():
            print(f"   {class_id}: {name}")
        
        print(f"\n🔄 Stage 1 → Stage 2 Mapping:")
        for s1_class, s2_class in self.stage1_to_stage2.items():
            s1_name = self.stage1_classes[s1_class]
            s2_name = self.stage2_classes[s2_class]
            print(f"   {s1_name:12s} → {s2_name}")
        
        print(f"\n🔄 Stage 2 → Stage 3 Mapping:")
        for s2_class, s3_class in self.stage2_to_stage3.items():
            s2_name = self.stage2_classes[s2_class]
            s3_name = self.stage3_classes[s3_class]
            print(f"   {s2_name:12s} → {s3_name}")


def adapt_model_for_new_stage(
    model: nn.Module,
    source_stage: int,
    target_stage: int,
    mapper: ProgressiveClassMapper,
    strategy: str = "linear_projection"
) -> nn.Module:
    """
    Adapt a trained model from one stage to another.
    
    Args:
        model: Trained model from source stage
        source_stage: Source stage number
        target_stage: Target stage number  
        mapper: Progressive class mapper
        strategy: Adaptation strategy ("linear_projection", "feature_reuse", "knowledge_distillation")
        
    Returns:
        Adapted model for target stage
    """
    
    if source_stage == 1 and target_stage == 2:
        source_classes = 24
        target_classes = 6
    elif source_stage == 2 and target_stage == 3:
        source_classes = 6
        target_classes = 6
    else:
        raise ValueError(f"Unsupported adaptation: Stage {source_stage} → Stage {target_stage}")
    
    # Find the final classification layer (segmentation head)
    final_layer = None
    final_layer_name = None
    
    # Look for the segmentation head's final layer first
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == source_classes:
            if 'seg_head' in name:  # Prioritize seg_head layers
                final_layer = module
                final_layer_name = name
                break
    
    # If not found in seg_head, search more broadly
    if final_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == source_classes:
                final_layer = module
                final_layer_name = name
                break
    
    if final_layer is None:
        raise ValueError(f"Could not find classification layer with {source_classes} classes")
    
    if strategy == "linear_projection":
        # Create new classification layer
        new_layer = nn.Conv2d(
            final_layer.in_channels,
            target_classes,
            kernel_size=final_layer.kernel_size,
            stride=final_layer.stride,
            padding=final_layer.padding,
            bias=final_layer.bias is not None
        )
        
        # Initialize with mapped weights
        mapping_matrix = mapper.get_mapping_matrix(source_stage, target_stage)
        
        with torch.no_grad():
            # Map weights: [target_classes, in_channels, k, k]
            old_weight = final_layer.weight  # [source_classes, in_channels, k, k]
            new_weight = torch.zeros_like(new_layer.weight)
            
            for target_class in range(target_classes):
                # Average weights from all source classes that map to this target class
                source_classes_for_target = (mapping_matrix[target_class] > 0).nonzero().squeeze()
                if source_classes_for_target.numel() > 0:
                    if source_classes_for_target.numel() == 1:
                        new_weight[target_class] = old_weight[source_classes_for_target]
                    else:
                        new_weight[target_class] = old_weight[source_classes_for_target].mean(dim=0)
            
            new_layer.weight.copy_(new_weight)
            
            # Map bias if present
            if final_layer.bias is not None:
                old_bias = final_layer.bias
                new_bias = torch.zeros_like(new_layer.bias)
                
                for target_class in range(target_classes):
                    source_classes_for_target = (mapping_matrix[target_class] > 0).nonzero().squeeze()
                    if source_classes_for_target.numel() > 0:
                        if source_classes_for_target.numel() == 1:
                            new_bias[target_class] = old_bias[source_classes_for_target]
                        else:
                            new_bias[target_class] = old_bias[source_classes_for_target].mean()
                
                new_layer.bias.copy_(new_bias)
        
        # Replace the layer in the model
        parent_module = model
        *path, layer_name = final_layer_name.split('.')
        for p in path:
            parent_module = getattr(parent_module, p)
        setattr(parent_module, layer_name, new_layer)
        
        print(f"✅ Model adapted: {source_classes} → {target_classes} classes")
        print(f"   Strategy: {strategy}")
        print(f"   Layer adapted: {final_layer_name}")
    
    return model


if __name__ == "__main__":
    # Test progressive mapping system
    mapper = ProgressiveClassMapper()
    mapper.print_mapping_summary()
    
    # Test mapping matrices
    print("\n🧪 Testing Mapping Matrices:")
    
    matrix_1_to_2 = mapper.get_mapping_matrix(1, 2)
    print(f"Stage 1→2 matrix shape: {matrix_1_to_2.shape}")
    
    matrix_2_to_3 = mapper.get_mapping_matrix(2, 3)
    print(f"Stage 2→3 matrix shape: {matrix_2_to_3.shape}")
    
    # Test prediction conversion
    dummy_preds = torch.randn(2, 24, 64, 64)  # Batch of Stage 1 predictions
    converted_preds = mapper.convert_predictions(dummy_preds, 1, 2)
    print(f"Predictions converted: {dummy_preds.shape} → {converted_preds.shape}")
    
    print("\n✅ Progressive mapping system test complete!")
