#!/usr/bin/env python3
"""
MobileNetV3-Based Edge Model for UAV Landing
===========================================

Implements the edge-optimized architecture from EDGE_OPTIMIZED_STRATEGY.md:
- MobileNetV3-Small backbone (2.5MB, optimized for mobile)
- Lightweight segmentation head for 6 landing classes
- <15-25ms inference on edge hardware
- Multi-stage progressive training compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from typing import Dict, Optional, Union
import warnings


# ==============================================================================
# LEGACY MODEL DEFINITION - DO NOT MODIFY
# This class represents the architecture that was used to save the old
# checkpoints (like stage2_best.pth). It is preserved for backward compatibility.
# ==============================================================================
class LegacyEdgeLandingNet(nn.Module):
    """
    This is the legacy architecture that matches your saved `stage2_best.pth`.
    Based on checkpoint analysis, here's the exact structure:
    - seg_head.0: Conv2d(576, 128, 3x3)
    - seg_head.1: BatchNorm2d(128)  
    - seg_head.4: Conv2d(128, 6, 1x1)
    - uncertainty_head.0: Conv2d(128, 64, 3x3)
    - uncertainty_head.1: BatchNorm2d(64)
    - uncertainty_head.3: Conv2d(64, 1, 1x1)
    """
    def __init__(self, num_classes: int = 6, backbone_pretrained: bool = True, use_uncertainty: bool = False, dropout: float=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty
        
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if backbone_pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        # Note: Keep the classifier as it exists in the checkpoint
        
        # Exact segmentation head structure from checkpoint
        self.seg_head = nn.ModuleList([
            nn.Conv2d(576, 128, kernel_size=3, padding=1, bias=False),  # seg_head.0
            nn.BatchNorm2d(128),                                        # seg_head.1
            None,  # seg_head.2 (doesn't exist)
            None,  # seg_head.3 (doesn't exist)
            nn.Conv2d(128, self.num_classes, kernel_size=1, bias=False) # seg_head.4
        ])

        # Exact uncertainty head structure from checkpoint
        if use_uncertainty:
            self.uncertainty_head = nn.ModuleList([
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),  # uncertainty_head.0
                nn.BatchNorm2d(64),                                        # uncertainty_head.1
                None,  # uncertainty_head.2 (doesn't exist)
                nn.Conv2d(64, 1, kernel_size=1, bias=False)               # uncertainty_head.3
            ])
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        input_shape = x.shape[-2:]
        features = self.backbone.features(x)  # [B, 576, H/16, W/16]
        
        # Apply segmentation head layers exactly as they were saved
        x_seg = self.seg_head[0](features)  # Conv: 576->128
        x_seg = self.seg_head[1](x_seg)     # BatchNorm
        x_seg = F.relu(x_seg, inplace=True) # ReLU (not saved as a parameter)
        seg_logits = self.seg_head[4](x_seg)  # Conv: 128->6 (final prediction)
        
        # Upsample to input size
        seg_logits = F.interpolate(seg_logits, size=input_shape, mode='bilinear', align_corners=False)

        if not self.use_uncertainty:
            return seg_logits
            
        # Apply uncertainty head
        x_unc = self.uncertainty_head[0](x_seg)  # Conv: 128->64
        x_unc = self.uncertainty_head[1](x_unc)  # BatchNorm  
        x_unc = F.relu(x_unc, inplace=True)      # ReLU (not saved as a parameter)
        uncertainty = self.uncertainty_head[3](x_unc)  # Conv: 64->1
        uncertainty = torch.sigmoid(uncertainty)  # Sigmoid (not saved as a parameter)
        
        # Upsample to input size
        uncertainty = F.interpolate(uncertainty, size=input_shape, mode='bilinear', align_corners=False)
        
        return {'main': seg_logits, 'uncertainty': uncertainty}


# ==============================================================================
# CURRENT (NEW) MODEL DEFINITION
# ==============================================================================
class EdgeLandingNet(nn.Module):
    """
    MobileNetV3-Small + Custom Segmentation Head for UAV Landing Detection.
    
    Architecture:
    - Backbone: MobileNetV3-Small (2.5MB, ImageNet pretrained)
    - Head: Lightweight segmentation decoder
    - Output: 6 landing classes
    - Speed: ~15-25ms inference on modern edge hardware
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        backbone_pretrained: bool = True,
        dropout: float = 0.2,
        use_uncertainty: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty
        
        # MobileNetV3-Small backbone
        if backbone_pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights)
        else:
            self.backbone = mobilenet_v3_small(weights=None)
        
        # Remove classifier head - we only need features
        self.backbone.classifier = nn.Identity()
        
        # Get feature dimensions
        # MobileNetV3-Small features output: [B, 576, H/16, W/16]
        backbone_out_channels = 576
        
        # Lightweight segmentation head
        self.seg_head = nn.Sequential(
            # Feature compression and upsampling
            nn.Conv2d(backbone_out_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            # Intermediate upsampling (16x -> 8x)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            # Further upsampling (8x -> 4x)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final upsampling to match input (4x -> 1x)
            nn.ConvTranspose2d(32, num_classes, 4, stride=4, padding=0)
        )
        
        # Uncertainty estimation head (optional)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(backbone_out_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(64, 1, 1),  # Single channel uncertainty
                nn.Sigmoid()  # Uncertainty in [0, 1]
            )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"🚁 EdgeLandingNet initialized:")
        print(f"   Backbone: MobileNetV3-Small ({'pretrained' if backbone_pretrained else 'random'})")
        print(f"   Classes: {num_classes}")
        print(f"   Uncertainty: {use_uncertainty}")
        print(f"   Parameters: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize segmentation head weights."""
        for module in [self.seg_head]:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        if hasattr(self, 'uncertainty_head'):
            for m in self.uncertainty_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        """
        features = self.backbone.features(x)  # [B, 576, H/16, W/16]
        seg_logits = self.seg_head(features)  # [B, num_classes, H, W]
        
        if not self.use_uncertainty:
            return seg_logits
        
        uncertainty = self.uncertainty_head(features)
        uncertainty = F.interpolate(
            uncertainty, 
            size=seg_logits.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return {
            'main': seg_logits,
            'uncertainty': uncertainty
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedEdgeLandingNet(nn.Module):
    """
    Enhanced edge-optimized landing detection network with multi-scale features.
    """
    def __init__(
        self, 
        num_classes: int = 6, 
        backbone_pretrained: bool = True,
        dropout: float = 0.2,
        use_uncertainty: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_uncertainty = use_uncertainty
        
        # MobileNetV3-Small backbone with multi-scale feature extraction
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if backbone_pretrained else None
        mobilenet = mobilenet_v3_small(weights=weights)
        
        # Extract feature stages for multi-scale processing
        self.stem = nn.Sequential(*list(mobilenet.features)[:2])
        self.stage1 = nn.Sequential(*list(mobilenet.features)[2:4])
        self.stage2 = nn.Sequential(*list(mobilenet.features)[4:7])
        self.stage3 = nn.Sequential(*list(mobilenet.features)[7:11])
        self.stage4 = nn.Sequential(*list(mobilenet.features)[11:])
        
        # Dynamically determine channel dimensions
        with torch.no_grad():
            test_x = torch.randn(1, 3, 512, 512)
            x3 = self.stage2(self.stage1(self.stem(test_x)))
            x4 = self.stage3(x3)
            x5 = self.stage4(x4)
        
        stage3_channels = x3.shape[1]
        stage4_channels = x4.shape[1]
        stage5_channels = x5.shape[1]
        total_channels = stage3_channels + stage4_channels + stage5_channels
        
        # Feature fusion layer
        self.fusion_conv = nn.Conv2d(total_channels, 256, 1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(256)
        self.fusion_relu = nn.ReLU(inplace=True)
        
        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, num_classes, 4, stride=2, padding=1),
        )
        
        # Uncertainty estimation head
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
        
        self._init_weights()
        print(f"🚁 EnhancedEdgeLandingNet initialized: Params: {self.count_parameters():,}")

    def _init_weights(self):
        for m in [self.fusion_conv, self.decoder, self.uncertainty_head if self.use_uncertainty else nn.Identity()]:
            for module in m.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Multi-scale feature extraction
        x3 = self.stage2(self.stage1(self.stem(x)))
        x4 = self.stage3(x3)
        x5 = self.stage4(x4)
        
        # Align features to same spatial resolution
        x3_up = F.interpolate(x3, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        x4_up = F.interpolate(x4, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        
        # Feature fusion
        fused = torch.cat([x5, x4_up, x3_up], dim=1)
        fused = self.fusion_relu(self.fusion_bn(self.fusion_conv(fused)))
        
        # Decode to segmentation map
        seg_logits = self.decoder(fused)
        
        if not self.use_uncertainty:
            return seg_logits
        
        uncertainty = self.uncertainty_head(fused)
        uncertainty = F.interpolate(uncertainty, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False)
        
        return {'main': seg_logits, 'uncertainty': uncertainty}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_edge_model(
    model_type: str = "standard",
    num_classes: int = 6,
    input_size: int = 512,
    use_uncertainty: bool = False,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to create edge-optimized models.
    
    Args:
        model_type: 'standard', 'enhanced', or 'legacy'
        num_classes: Number of segmentation classes
        input_size: Input image size (not used in current implementation)
        use_uncertainty: Whether to include uncertainty estimation
        pretrained: Whether to use ImageNet pretrained backbone
    
    Returns:
        Configured model instance
    """
    
    if model_type == "standard":
        model = EdgeLandingNet(
            num_classes=num_classes,
            backbone_pretrained=pretrained,
            use_uncertainty=use_uncertainty
        )
    elif model_type == "legacy":
        print("   Instantiating LEGACY EdgeLandingNet for backward compatibility.")
        model = LegacyEdgeLandingNet(
            num_classes=num_classes,
            backbone_pretrained=pretrained,
            use_uncertainty=use_uncertainty
        )
    elif model_type == "enhanced":
        model = EnhancedEdgeLandingNet(
            num_classes=num_classes,
            backbone_pretrained=pretrained,
            use_uncertainty=use_uncertainty
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'standard', 'enhanced', or 'legacy'")
    
    return model


# Additional utility functions
def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 512, 512)) -> str:
    """Generate a summary of model architecture and parameters."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Summary:
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}
- Input Size: {input_size}
"""
    
    return summary


def optimize_for_mobile(model: nn.Module) -> nn.Module:
    """Apply mobile-specific optimizations to the model."""
    
    # Convert BatchNorm to more mobile-friendly operations if needed
    # This is a placeholder for future mobile optimizations
    warnings.warn("Mobile optimization not yet implemented", UserWarning)
    
    return model


if __name__ == "__main__":
    # Test model creation and benchmarking
    print("🚁 Testing Edge Models...")
    
    # Test standard model
    print("\n=== Standard EdgeLandingNet ===")
    model_std = create_edge_model(
        model_type="standard",
        num_classes=6,
        use_uncertainty=True,
        pretrained=True
    )
    
    # Test enhanced model
    print("\n=== Enhanced EdgeLandingNet ===")
    model_enh = create_edge_model(
        model_type="enhanced", 
        num_classes=6,
        use_uncertainty=True,
        pretrained=True
    )
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    dummy_input = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        output_std = model_std(dummy_input)
        output_enh = model_enh(dummy_input)
    
    print(f"Standard output shapes:")
    print(f"   Main: {output_std['main'].shape}")
    print(f"   Uncertainty: {output_std['uncertainty'].shape}")
    
    print(f"Enhanced output shapes:")
    print(f"   Main: {output_enh['main'].shape}")
    print(f"   Uncertainty: {output_enh['uncertainty'].shape}")
    
    # Benchmark if CUDA available
    if torch.cuda.is_available():
        print(f"\n⚡ Speed Benchmarks:")
        benchmark_model_speed(model_std, input_size=512, device='cuda')
        benchmark_model_speed(model_enh, input_size=512, device='cuda')
    else:
        print(f"\n⚠️  CUDA not available, skipping speed benchmark") 