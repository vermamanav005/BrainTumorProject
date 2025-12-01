"""
Model Architectures for Brain Tumor Classification

This module contains all model definitions:
- TumorNetLite: Novel lightweight architecture with SCTA, APF, PFR
- DMFNet: Dual-stream feature fusion network
- Baseline models: ResNet50, EfficientNet-B0, MobileNet-V2/V3

All models follow consistent interface for easy experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List


###############################################################################
# TUMORNET-LITE COMPONENTS
###############################################################################

class SpatialChannelTumorAttention(nn.Module):
    """
    Spatial-Channel Tumor Attention (SCTA)
    
    Dual attention mechanism that enhances tumor-specific features:
    1. Channel attention: Learns "what" features are important
    2. Spatial attention: Learns "where" tumors are located
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for efficiency (default: 16)
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SpatialChannelTumorAttention, self).__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Channel Attention
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)
        
        avg_out = self.channel_fc(avg_pool)
        max_out = self.channel_fc(max_pool)
        
        channel_att = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        x = x * channel_att  # Apply channel attention
        
        # Spatial Attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        
        spatial_att = self.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_att  # Apply spatial attention
        
        return x


class AsymmetricPyramidFusion(nn.Module):
    """
    Asymmetric Pyramid Fusion (APF)
    
    Efficient multi-scale feature integration with learnable hierarchical weights.
    Combines features from different scales using asymmetric convolutions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scales: List of dilation rates for multi-scale processing
    """
    def __init__(self, in_channels: int, out_channels: int, scales: List[int] = [1, 2, 4]):
        super(AsymmetricPyramidFusion, self).__init__()
        
        self.scales = scales
        self.branches = nn.ModuleList()
        
        for scale in scales:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(scales), 
                         kernel_size=3, padding=scale, dilation=scale, bias=False),
                nn.BatchNorm2d(out_channels // len(scales)),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(scales)))
        
        # 1x1 conv for channel alignment
        self.channel_align = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Process each scale
        features = []
        for i, branch in enumerate(self.branches):
            feat = branch(x) * self.fusion_weights[i]
            features.append(feat)
        
        # Concatenate and align
        out = torch.cat(features, dim=1)
        out = self.channel_align(out)
        
        return out


class ProgressiveFeatureRefinement(nn.Module):
    """
    Progressive Feature Refinement (PFR)
    
    Multi-receptive field processing for enhanced feature discrimination.
    Uses progressive convolutions with increasing receptive fields.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_stages: Number of refinement stages (default: 3)
    """
    def __init__(self, in_channels: int, out_channels: int, num_stages: int = 3):
        super(ProgressiveFeatureRefinement, self).__init__()
        
        self.stages = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_stages):
            stage_channels = out_channels if i == num_stages - 1 else in_channels
            
            stage = nn.Sequential(
                nn.Conv2d(current_channels, stage_channels, 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stage_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(stage_channels, stage_channels,
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stage_channels),
                nn.ReLU(inplace=True)
            )
            self.stages.append(stage)
            current_channels = stage_channels
        
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.residual(x)
        
        out = x
        for stage in self.stages:
            out = stage(out)
        
        out = out + identity
        return F.relu(out)


###############################################################################
# TUMORNET-LITE MAIN MODEL
###############################################################################

class TumorNetLite(nn.Module):
    """
    TumorNet-Lite: Lightweight Deep Learning Framework for Brain Tumor Classification
    
    Novel architecture combining:
    - Spatial-Channel Tumor Attention (SCTA)
    - Asymmetric Pyramid Fusion (APF)
    - Progressive Feature Refinement (PFR)
    - Uncertainty-Aware Classification
    
    Args:
        num_classes: Number of output classes (default: 4)
        pretrained: Use pretrained backbone (default: False)
        in_channels: Number of input channels (default: 3 for RGB)
        base_channels: Base number of channels (default: 64)
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = False, 
                 in_channels: int = 3, base_channels: int = 64):
        super(TumorNetLite, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: Basic feature extraction
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.scta1 = SpatialChannelTumorAttention(base_channels * 2)
        
        # Stage 2: Asymmetric pyramid fusion
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            AsymmetricPyramidFusion(base_channels * 2, base_channels * 4, scales=[1, 2, 4])
        )
        self.scta2 = SpatialChannelTumorAttention(base_channels * 4)
        
        # Stage 3: Progressive feature refinement
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ProgressiveFeatureRefinement(base_channels * 4, base_channels * 8, num_stages=3)
        )
        self.scta3 = SpatialChannelTumorAttention(base_channels * 8)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_channels * 8 * 2, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial features
        x = self.conv1(x)
        
        # Stage 1 with SCTA
        x = self.stage1(x)
        x = self.scta1(x)
        
        # Stage 2 with APF and SCTA
        x = self.stage2(x)
        x = self.scta2(x)
        
        # Stage 3 with PFR and SCTA
        x = self.stage3(x)
        x = self.scta3(x)
        
        # Global pooling (avg + max for uncertainty awareness)
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


###############################################################################
# DMFNET (Dual-stream Multi-scale Feature Network)
###############################################################################

class DepthAttentionModule(nn.Module):
    """
    Depth Attention Module for DMFNet
    
    Learns importance weights for different feature depths.
    """
    def __init__(self, in_channels: int):
        super(DepthAttentionModule, self).__init__()
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)
        return x * att


class DMFNet(nn.Module):
    """
    DMFNet: Dual-stream Multi-scale Feature Network
    
    Two-stream architecture:
    - Shallow stream: Low-level features (edges, textures)
    - Deep stream: High-level features (semantic information)
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained MobileNetV3 backbone
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super(DMFNet, self).__init__()
        
        # Backbone: MobileNetV3-Small
        if pretrained:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            backbone = models.mobilenet_v3_small(weights=None)
        
        self.features = backbone.features
        
        # Shallow stream (early layers)
        self.shallow_stream = nn.Sequential(*list(self.features[:4]))
        self.shallow_attention = DepthAttentionModule(24)
        
        # Deep stream (later layers)
        self.deep_stream = nn.Sequential(*list(self.features[4:]))
        self.deep_attention = DepthAttentionModule(576)
        
        # Fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(24 + 576, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Shallow stream
        shallow_feat = self.shallow_stream(x)
        shallow_feat = self.shallow_attention(shallow_feat)
        
        # Deep stream
        deep_feat = self.deep_stream(shallow_feat)
        deep_feat = self.deep_attention(deep_feat)
        
        # Upsample shallow features to match deep features
        shallow_feat_up = F.interpolate(shallow_feat, size=deep_feat.shape[2:], 
                                        mode='bilinear', align_corners=False)
        
        # Fusion
        fused = torch.cat([shallow_feat_up, deep_feat], dim=1)
        fused = self.fusion_conv(fused)
        
        # Global pooling and classification
        x = self.global_pool(fused).view(fused.size(0), -1)
        x = self.classifier(x)
        
        return x


###############################################################################
# BASELINE MODELS
###############################################################################

def get_resnet50(num_classes: int = 4, pretrained: bool = True):
    """
    ResNet-50 baseline model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        Modified ResNet-50 model
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)
    
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_efficientnet_b0(num_classes: int = 4, pretrained: bool = True):
    """
    EfficientNet-B0 baseline model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        Modified EfficientNet-B0 model
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_mobilenet_v2(num_classes: int = 4, pretrained: bool = True):
    """
    MobileNet-V2 baseline model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        Modified MobileNet-V2 model
    """
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        model = models.mobilenet_v2(weights=None)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_mobilenet_v3_small(num_classes: int = 4, pretrained: bool = True):
    """
    MobileNet-V3-Small baseline model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        Modified MobileNet-V3-Small model
    """
    if pretrained:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        model = models.mobilenet_v3_small(weights=None)
    
    # Replace classifier
    in_features = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(1024, num_classes)
    )
    
    return model


###############################################################################
# MODEL FACTORY
###############################################################################

def get_model(model_name: str, num_classes: int = 4, pretrained: bool = False, **kwargs):
    """
    Factory function to get any model by name.
    
    Args:
        model_name: Name of model ('tumornet_lite', 'dmfnet', 'resnet50', etc.)
        num_classes: Number of output classes
        pretrained: Use pretrained weights (where applicable)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Requested model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    if model_name == 'tumornet_lite':
        return TumorNetLite(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'dmfnet':
        return DMFNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet50':
        return get_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        return get_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenet_v2':
        return get_mobilenet_v2(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenet_v3_small':
        return get_mobilenet_v3_small(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available models: tumornet_lite, dmfnet, resnet50, efficientnet_b0, "
            f"mobilenet_v2, mobilenet_v3_small"
        )


def count_parameters(model: nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str):
    """
    Print comprehensive model summary.
    
    Args:
        model: PyTorch model
        model_name: Name of model
    """
    total_params = count_parameters(model)
    
    print("\n" + "="*70)
    print(f"MODEL SUMMARY: {model_name}")
    print("="*70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Size (MB): {total_params * 4 / (1024**2):.2f}")  # Assuming float32
    print("="*70 + "\n")


if __name__ == "__main__":
    print("="*70)
    print("BRAIN TUMOR CLASSIFICATION - MODEL ARCHITECTURES")
    print("="*70)
    
    # Test all models
    print("\nTesting model creation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_to_test = [
        'tumornet_lite',
        'dmfnet',
        'resnet50',
        'efficientnet_b0',
        'mobilenet_v2',
        'mobilenet_v3_small'
    ]
    
    for model_name in models_to_test:
        try:
            model = get_model(model_name, num_classes=4, pretrained=False)
            model = model.to(device)
            print_model_summary(model, model_name)
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            output = model(dummy_input)
            assert output.shape == (2, 4), f"Output shape mismatch for {model_name}"
            print(f"✓ {model_name}: Forward pass successful")
            
        except Exception as e:
            print(f"✗ {model_name}: Error - {str(e)}")
    
    print("\n" + "="*70)
    print("Model testing complete!")
    print("="*70)
