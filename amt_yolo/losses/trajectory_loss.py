"""
trajectory_loss.py — Trajectory Prediction Loss Functions
==========================================================
Combined loss for trajectory prediction head in AMT-YOLO.

Components:
    1. SmoothL1Loss  — Box coordinate regression (robust to outliers)
    2. ADELoss       — Average Displacement Error (spatial accuracy)
    3. ConfidenceLoss — BCE per-step confidence calibration

Final loss:
    L_traj = lambda_smooth * L_smooth_l1
           + lambda_ade   * L_ade
           + lambda_conf  * L_conf

Reference:
    Average Displacement Error = mean Euclidean distance between
    predicted and GT center points across all time steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AverageFinalDisplacementLoss(nn.Module):
    """
    Computes both Average Displacement Error (ADE) and
    Final Displacement Error (FDE) as training losses.

    ADE: mean L2 distance over all predicted time steps.
    FDE: L2 distance only at the final predicted step.
    Both operate on (cx, cy) coordinates.
    """

    def __init__(self, fde_weight: float = 0.5):
        """
        Args:
            fde_weight: Additional weight on the final step error (0 = ADE only)
        """
        super().__init__()
        self.fde_weight = fde_weight

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes:  [N, horizon, 4] predicted future boxes (cx, cy, w, h)
            gt_boxes:    [N, horizon, 4] ground truth future boxes
            valid_mask:  [N, horizon] bool — True where GT is available

        Returns:
            loss: scalar
        """
        if valid_mask.sum() == 0:
            # Return zero connected to computational graph so backward still works
            return pred_boxes.sum() * 0.0

        # Compute L2 displacement on (cx, cy) only
        pred_centers = pred_boxes[..., :2]   # [N, H, 2]
        gt_centers   = gt_boxes[..., :2]     # [N, H, 2]

        l2_per_step = torch.norm(pred_centers - gt_centers, dim=-1)  # [N, H]

        # Apply validity mask
        valid_l2 = l2_per_step * valid_mask.float()
        n_valid  = valid_mask.float().sum(dim=-1).clamp(min=1)  # [N]

        # ADE: mean over all valid steps per object, then mean over objects
        ade = (valid_l2.sum(dim=-1) / n_valid).mean()

        # FDE: error at the LAST valid step for each object
        # Find the last True index in each row
        last_valid_idx = (valid_mask.long() * torch.arange(
            valid_mask.shape[-1], device=pred_boxes.device
        )).argmax(dim=-1)  # [N]

        fde_per_obj = l2_per_step[torch.arange(l2_per_step.shape[0]), last_valid_idx]
        fde = fde_per_obj.mean()

        return ade + self.fde_weight * fde


class TrajectoryLoss(nn.Module):
    """
    Combined trajectory prediction loss for AMT-YOLO.

    Args:
        lambda_smooth:  Weight for SmoothL1 regression loss
        lambda_ade:     Weight for ADE/FDE displacement loss
        lambda_conf:    Weight for per-step confidence calibration loss
        fde_weight:     Additional weight on Final Displacement Error
    """

    def __init__(
        self,
        lambda_smooth: float = 0.5,
        lambda_ade:    float = 0.3,
        lambda_conf:   float = 0.2,
        fde_weight:    float = 0.5,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_ade    = lambda_ade
        self.lambda_conf   = lambda_conf

        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.ade_loss  = AverageFinalDisplacementLoss(fde_weight=fde_weight)
        self.bce       = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        gt_boxes:   torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_dict:   Output of TrajectoryPredictionHead, containing:
                         - 'future_boxes':  [N, horizon, 4]
                         - 'confidences':   [N, horizon, 1]
            gt_boxes:    [N, horizon, 4] ground truth future boxes (normalized cx,cy,w,h)
            valid_mask:  [N, horizon] boolean mask (True = GT available)

        Returns:
            dict with keys: 'total', 'smooth_l1', 'ade', 'conf'
        """
        future_boxes  = pred_dict["future_boxes"]   # [N, H, 4]
        confidences   = pred_dict["confidences"]    # [N, H, 1]
        horizon       = future_boxes.shape[1]

        device = future_boxes.device

        if future_boxes.shape[0] == 0:
            # Return graph-connected zeros to keep backward connected to model params
            zero = future_boxes.sum() * 0.0
            return {"total": zero, "smooth_l1": zero, "ade": zero, "conf": zero}

        # ── 1. SmoothL1 Box Regression Loss ──────────────────────────────
        # Only computed on valid GT steps
        mask_4d = valid_mask.unsqueeze(-1).expand_as(future_boxes).float()  # [N, H, 4]
        raw_smooth = self.smooth_l1(future_boxes, gt_boxes) * mask_4d
        n_valid = mask_4d.sum().clamp(min=1)
        loss_smooth = raw_smooth.sum() / n_valid

        # ── 2. ADE / FDE Displacement Loss ───────────────────────────────
        loss_ade = self.ade_loss(future_boxes, gt_boxes, valid_mask)

        # ── 3. Confidence Calibration Loss ───────────────────────────────
        # Treat GT-valid steps as positive (conf target=1.0), occluded as 0.0
        conf_target = valid_mask.float().unsqueeze(-1)  # [N, H, 1]
        loss_conf   = self.bce(confidences, conf_target).mean()

        # ── 4. Combined Loss ──────────────────────────────────────────────
        total = (
            self.lambda_smooth * loss_smooth
            + self.lambda_ade  * loss_ade
            + self.lambda_conf * loss_conf
        )

        return {
            "total":     total,
            "smooth_l1": loss_smooth.detach(),
            "ade":       loss_ade.detach(),
            "conf":      loss_conf.detach(),
        }
