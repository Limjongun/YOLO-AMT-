"""
adaptive_resolution.py — Adaptive Resolution Router
====================================================
Dynamically selects input resolution based on scene complexity
to balance compute efficiency and detection accuracy.

Supported resolutions (training): 640, 768
Supported resolutions (inference): 640, 768, 1024

Scene complexity estimation:
    - Edge density (fast, heuristic-based)
    - Object count estimation (lightweight CNN, optional)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class EdgeDensityEstimator(nn.Module):
    """
    Lightweight edge-density based complexity estimator.
    Uses Sobel filter — no learnable parameters, zero overhead.

    Returns a scalar in [0, 1] representing scene complexity.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels (fixed, not learned)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frame [B, 3, H, W] in [0, 1]
        Returns:
            complexity: Scalar [B] in [0, 1]
        """
        # Convert to grayscale
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # [B, H, W]
        gray = gray.unsqueeze(1)  # [B, 1, H, W]

        # Apply Sobel
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2)  # [B, 1, H, W]

        # Normalize edge density to [0, 1]
        complexity = magnitude.mean(dim=[1, 2, 3])  # [B]
        complexity = (complexity / complexity.max().clamp(min=1e-6)).clamp(0, 1)
        return complexity


class LearnedComplexityEstimator(nn.Module):
    """
    Optional: Lightweight CNN complexity estimator (learned).
    More accurate but adds ~0.1ms overhead per frame.
    Trained with supervision from FPS drop signal.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # [B]


class AdaptiveResolutionRouter(nn.Module):
    """
    Adaptive Resolution Router for AMT-YOLO.

    Selects the optimal input resolution for each frame based on
    estimated scene complexity. Simpler scenes → lower resolution (faster FPS),
    complex scenes → higher resolution (better accuracy for small objects).

    Args:
        resolutions: List of candidate resolutions, e.g. [640, 768, 1024]
        thresholds:  Complexity thresholds for each step. len = len(resolutions) - 1
        estimator:   'edge' (heuristic, default) or 'learned'
        training_max_res: Maximum resolution used during training (VRAM constraint)
    """

    def __init__(
        self,
        resolutions: List[int] = [640, 768, 1024],
        thresholds: List[float] = [0.3, 0.65],
        estimator: str = "edge",
        training_max_res: int = 768,  # RTX 4050 6GB constraint
    ):
        super().__init__()
        assert len(thresholds) == len(resolutions) - 1, \
            f"Need {len(resolutions)-1} thresholds for {len(resolutions)} resolutions"

        self.resolutions = sorted(resolutions)
        self.thresholds = thresholds
        self.training_max_res = training_max_res

        # Complexity estimator
        if estimator == "edge":
            self.estimator = EdgeDensityEstimator()
        elif estimator == "learned":
            self.estimator = LearnedComplexityEstimator()
        else:
            raise ValueError(f"Unknown estimator: {estimator}. Use 'edge' or 'learned'.")

        self._selected_resolution: int = resolutions[0]  # Track for logging

    def select_resolution(self, complexity: torch.Tensor) -> int:
        """
        Select resolution from complexity score.

        Args:
            complexity: Batch complexity [B], takes mean over batch
        Returns:
            Selected resolution (int)
        """
        score = complexity.mean().item()
        selected = self.resolutions[0]
        for i, threshold in enumerate(self.thresholds):
            if score >= threshold:
                selected = self.resolutions[i + 1]
            else:
                break

        # Enforce VRAM constraint during training
        if self.training:
            selected = min(selected, self.training_max_res)

        self._selected_resolution = selected
        return selected

    def resize_frame(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        """Resize frame to selected square resolution."""
        if x.shape[-1] == resolution and x.shape[-2] == resolution:
            return x
        return F.interpolate(
            x,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, float]:
        """
        Args:
            x: Input frame [B, 3, H, W], normalized [0, 1]
        Returns:
            resized_x: Resized frame at selected resolution
            resolution: Selected resolution int
            complexity: Scene complexity score (for logging)
        """
        complexity = self.estimator(x)
        resolution = self.select_resolution(complexity)
        resized_x = self.resize_frame(x, resolution)
        return resized_x, resolution, complexity.mean().item()

    @property
    def last_resolution(self) -> int:
        """Return the last selected resolution (for logging)."""
        return self._selected_resolution
