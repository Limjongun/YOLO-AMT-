"""
detection_loss.py — Detection Loss Wrapper for AMT-YOLO
========================================================
Wraps the Ultralytics v8DetectionLoss and extends it with
tracking embedding regularization for AMT-YOLO's identity module.

Why wrap Ultralytics loss instead of writing from scratch?
    - Ultralytics v8DetectionLoss uses CIoU + DFL + BCE (best-in-class)
    - Rewriting it would introduce bugs and is not novel contribution
    - Our contribution is the Temporal Memory + Trajectory prediction
    - We extend with EmbeddingVarianceLoss for tracking stability

Loss Components:
    L_det = L_box (CIoU) + L_dfl (Distribution Focal) + L_cls (BCE)
    L_emb = EmbeddingVarianceLoss (optional, keeps embeddings stable)

    L_total = w_det * L_det + w_emb * L_emb
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EmbeddingVarianceLoss(nn.Module):
    """
    Encourages tracking embeddings to be compact within a sequence.
    Penalizes high variance in embedding space across frames,
    which would indicate unstable identity representations.

    This is a soft regularization — it doesn't enforce fixed IDs,
    just keeps the embedding distribution stable.
    """

    def __init__(self, target_variance: float = 0.1):
        """
        Args:
            target_variance: The maximum allowed variance per embedding dimension
        """
        super().__init__()
        self.target_variance = target_variance

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] tracking embeddings from the model

        Returns:
            variance regularization loss (scalar)
        """
        if embeddings is None or embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device if embeddings is not None else 'cpu')

        # Variance across batch dimension per embedding channel
        var = embeddings.var(dim=0)  # [D]
        # Penalize excess variance beyond target
        excess = F.relu(var - self.target_variance)
        return excess.mean()


class AMTDetectionLoss(nn.Module):
    """
    Detection loss wrapper for AMT-YOLO.

    Uses Ultralytics' native v8DetectionLoss internally for the
    detection head, then adds embedding regularization.

    Args:
        yolo_model:       The Ultralytics DetectionModel inside AMTYOLO.backbone.yolo_model
        embedding_weight: Weight for embedding variance regularization
    """

    def __init__(
        self,
        yolo_model,
        embedding_weight: float = 0.05,
    ):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.emb_loss = EmbeddingVarianceLoss()

        # Initialize Ultralytics native loss
        # This handles CIoU, DFL, BCE classification loss
        self._init_ultralytics_loss(yolo_model)

    def _init_ultralytics_loss(self, yolo_model):
        """Import and initialize Ultralytics v8DetectionLoss."""
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            from ultralytics.utils import DEFAULT_CFG
        
            # Ensure model.args has all default hyperparameters (like hyp.box, hyp.cls)
            yolo_model.args = DEFAULT_CFG
            
            self.v8_loss = v8DetectionLoss(yolo_model)
        except ImportError as e:
            raise ImportError(
                f"Ultralytics loss not available: {e}. "
                "Ensure ultralytics >= 8.2.0 is installed."
            )

    def forward(
        self,
        preds,
        targets: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds:       Raw predictions from YOLO detection head
            targets:     [N, 6] ground truth boxes (batch_idx, class, cx, cy, w, h)
            embeddings:  [B, D] tracking embeddings from AMT-YOLO (optional)

        Returns:
            dict with keys: 'total', 'box', 'cls', 'dfl', 'emb'
        """
        # Compute Ultralytics detection losses (box + cls + dfl)
        try:
            # Ultralytics v8DetectionLoss expects a dictionary, not a tensor
            batch_dict = {
                'batch_idx': targets[:, 0],
                'cls': targets[:, 1],
                'bboxes': targets[:, 2:]
            }
            loss_tuple, loss_items = self.v8_loss(preds, batch_dict)
            # Ultralytics returns (total, [box, cls, dfl]) or (loss_vector, ...)
            det_total = loss_tuple.sum() if len(loss_tuple.shape) > 0 else loss_tuple
            box_loss  = loss_items[0]
            cls_loss  = loss_items[1]
            dfl_loss  = loss_items[2]
        except Exception as e:
            print(f"Error computing Ultralytics loss: {e}")
            raise e

        # Embedding regularization
        if embeddings is not None and self.embedding_weight > 0:
            emb_loss = self.emb_loss(embeddings)
            total = det_total + self.embedding_weight * emb_loss
        else:
            emb_loss = torch.tensor(0.0)
            total = det_total

        return {
            "total": total,
            "box":   box_loss,
            "cls":   cls_loss,
            "dfl":   dfl_loss,
            "emb":   emb_loss,
        }
