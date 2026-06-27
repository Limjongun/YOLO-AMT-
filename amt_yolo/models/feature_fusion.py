"""
feature_fusion.py — Adaptive Feature Fusion Neck
=================================================
Fuses multi-scale features from YOLOv8 backbone with
adaptive attention weighting.

Builds on top of YOLOv8's PAN-FPN neck with:
    - Channel attention (SE block) per scale
    - Spatial attention for small object enhancement
    - Adaptive weight learning across scales
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)               # [B, C, 1, 1]
        scale = self.fc(scale)             # [B, C]
        scale = scale.view(*scale.shape, 1, 1)  # [B, C, 1, 1]
        return x * scale


class SpatialAttention(nn.Module):
    """Lightweight spatial attention for small object enhancement."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        max_ = x.max(dim=1, keepdim=True).values  # [B, 1, H, W]
        combined = torch.cat([avg, max_], dim=1)   # [B, 2, H, W]
        attention = torch.sigmoid(self.conv(combined))
        return x * attention


class AdaptiveScaleWeight(nn.Module):
    """
    Learns adaptive weights for feature fusion across scales.
    Replaces fixed equal weighting with learned importance.
    """

    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_scales))

    def forward(self) -> torch.Tensor:
        return F.softmax(self.weights, dim=0)  # [num_scales]


class AdaptiveFeatureFusionNeck(nn.Module):
    """
    Adaptive Feature Fusion Neck for AMT-YOLO.
    Enhances YOLOv8's PAN-FPN neck with attention-based adaptive fusion.

    Args:
        in_channels_list: List of channel counts per scale [P3, P4, P5]
        use_se_attention:     Enable channel attention (SE block)
        use_spatial_attention: Enable spatial attention
        use_adaptive_weights:  Enable learnable scale weights
    """

    def __init__(
        self,
        in_channels_list: List[int] = [128, 256, 512],
        use_se_attention: bool = True,
        use_spatial_attention: bool = True,
        use_adaptive_weights: bool = True,
    ):
        super().__init__()
        self.num_scales = len(in_channels_list)

        # Channel attention per scale
        self.se_blocks = nn.ModuleList([
            SEBlock(c) for c in in_channels_list
        ]) if use_se_attention else None

        # Spatial attention per scale
        self.spatial_attn = nn.ModuleList([
            SpatialAttention() for _ in in_channels_list
        ]) if use_spatial_attention else None

        # Adaptive scale weights
        self.scale_weights = AdaptiveScaleWeight(
            self.num_scales
        ) if use_adaptive_weights else None

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps [P3, P4, P5]
                      Each: [B, C_i, H_i, W_i]
        Returns:
            Enhanced feature maps (same shapes as input)
        """
        assert len(features) == self.num_scales

        # Get adaptive scale weights
        weights = self.scale_weights() if self.scale_weights is not None else None

        enhanced = []
        for i, feat in enumerate(features):
            f = feat

            # Channel attention
            if self.se_blocks is not None:
                f = self.se_blocks[i](f)

            # Spatial attention
            if self.spatial_attn is not None:
                f = self.spatial_attn[i](f)

            # Apply scale weight
            if weights is not None:
                f = f * weights[i]

            # Residual
            enhanced.append(f + feat)

        return enhanced
