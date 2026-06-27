"""
trajectory_head.py — Trajectory Prediction Head
================================================
Predicts future object positions (5–10 frames) using
tracked object embeddings + temporal memory features.

Architecture: GRU + MLP decoder
Input:  object embedding + memory embedding
Output: future center points, future bounding boxes, motion vectors

Loss: Smooth L1 + MSE + ADE/FDE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MotionEncoder(nn.Module):
    """
    Encodes object tracking embedding + memory embedding
    into a unified motion representation.
    """

    def __init__(self, obj_embed_dim: int = 128, mem_embed_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(obj_embed_dim + mem_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, obj_embed: torch.Tensor, mem_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_embed: Current object embedding [N, obj_embed_dim]
            mem_embed: Memory feature (pooled) [N, mem_embed_dim]
        Returns:
            motion_feat: [N, out_dim]
        """
        combined = torch.cat([obj_embed, mem_embed], dim=-1)
        return self.proj(combined)


class TrajectoryDecoder(nn.Module):
    """
    GRU-based multi-step trajectory decoder.
    Decodes motion features into future positions step by step.

    Args:
        input_dim:  Motion feature dimension
        hidden_dim: GRU hidden dimension
        horizon:    Number of future frames to predict (5-10)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # Initial hidden state projection
        self.h0_proj = nn.Linear(input_dim, hidden_dim)

        # GRU decoder cell
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

        # Output heads (per step)
        # Predict: (dx, dy, dw, dh) relative offsets
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # (cx, cy, w, h) offsets
        )

        # Confidence head: how confident are we in this future prediction
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        motion_feat: torch.Tensor,
        current_box: torch.Tensor,
        teacher_boxes: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            motion_feat:         [N, input_dim] — encoded motion
            current_box:         [N, 4] — current box (cx, cy, w, h) normalized
            teacher_boxes:       [N, horizon, 4] — GT future boxes (for training)
            teacher_forcing_ratio: Probability of using GT during training (0-1)

        Returns:
            dict with:
                'future_boxes':   [N, horizon, 4] predicted future boxes
                'motion_vectors': [N, horizon, 2] (dx, dy) per step
                'confidences':    [N, horizon, 1] confidence per step
        """
        N = motion_feat.shape[0]
        h = torch.tanh(self.h0_proj(motion_feat))  # [N, hidden_dim]

        pred_boxes = []
        pred_offsets = []
        pred_confs = []

        prev_box = current_box  # [N, 4]
        inp = motion_feat       # [N, input_dim]

        for t in range(self.horizon):
            h = self.gru_cell(inp, h)  # [N, hidden_dim]

            # Predict offset from previous box
            offset = self.offset_head(h)  # [N, 4]
            conf = self.conf_head(h)      # [N, 1]

            # Apply offset
            pred_box = prev_box + offset  # [N, 4]
            pred_boxes.append(pred_box)
            pred_offsets.append(offset[:, :2])  # (dx, dy) only for motion vector
            pred_confs.append(conf)

            # Teacher forcing during training
            if (
                teacher_boxes is not None
                and self.training
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                prev_box = teacher_boxes[:, t, :]
            else:
                prev_box = pred_box.detach()

            # Use predicted box as next input context (concatenated)
            inp = torch.cat([motion_feat, h], dim=-1)
            # Resize inp back to input_dim
            inp = motion_feat  # Simplified: reuse motion_feat each step

        return {
            "future_boxes": torch.stack(pred_boxes, dim=1),    # [N, H, 4]
            "motion_vectors": torch.stack(pred_offsets, dim=1), # [N, H, 2]
            "confidences": torch.stack(pred_confs, dim=1),      # [N, H, 1]
        }


class TrajectoryPredictionHead(nn.Module):
    """
    Full Trajectory Prediction Head for AMT-YOLO.
    Combines MotionEncoder + TrajectoryDecoder.

    Integrates with detection head tracking embeddings and
    temporal memory module outputs.

    Args:
        obj_embed_dim:   Object tracking embedding dim (from detection head)
        mem_embed_dim:   Memory embedding dim (from TemporalMemoryModule)
        hidden_dim:      Internal GRU hidden dimension
        horizon:         Prediction horizon (5–10 frames, default 5)
        teacher_forcing: Teacher forcing ratio during training (0.5 default)
    """

    def __init__(
        self,
        obj_embed_dim: int = 128,
        mem_embed_dim: int = 128,
        hidden_dim: int = 128,
        horizon: int = 5,
        teacher_forcing: float = 0.5,
    ):
        super().__init__()
        self.horizon = horizon
        self.teacher_forcing = teacher_forcing

        # Encode object + memory into motion representation
        self.encoder = MotionEncoder(
            obj_embed_dim=obj_embed_dim,
            mem_embed_dim=mem_embed_dim,
            out_dim=256,
        )

        # Multi-step trajectory decoder
        self.decoder = TrajectoryDecoder(
            input_dim=256,
            hidden_dim=hidden_dim,
            horizon=horizon,
        )

    def forward(
        self,
        obj_embed: torch.Tensor,
        mem_embed: torch.Tensor,
        current_boxes: torch.Tensor,
        teacher_boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            obj_embed:    [N, obj_embed_dim] — tracking embedding per object
            mem_embed:    [N, mem_embed_dim] — pooled memory feature per object
            current_boxes: [N, 4] — current detection boxes (cx, cy, w, h)
            teacher_boxes: [N, horizon, 4] — GT future boxes (training only)

        Returns:
            Dict with 'future_boxes', 'motion_vectors', 'confidences'
        """
        if obj_embed.shape[0] == 0:
            # No detections in this frame
            return {
                "future_boxes": torch.zeros(0, self.horizon, 4, device=obj_embed.device),
                "motion_vectors": torch.zeros(0, self.horizon, 2, device=obj_embed.device),
                "confidences": torch.zeros(0, self.horizon, 1, device=obj_embed.device),
            }

        motion_feat = self.encoder(obj_embed, mem_embed)
        return self.decoder(
            motion_feat,
            current_boxes,
            teacher_boxes=teacher_boxes,
            teacher_forcing_ratio=self.teacher_forcing if self.training else 0.0,
        )
