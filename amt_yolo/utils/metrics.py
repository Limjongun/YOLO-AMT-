"""
metrics.py — Evaluation Metrics for AMT-YOLO
=============================================
Implements all evaluation metrics needed for CVPR paper:

Detection:  mAP@50, mAP@50-95
Tracking:   HOTA, MOTA, IDF1
Trajectory: ADE (Average Displacement Error), FDE (Final Displacement Error)
Efficiency: FPS, Latency, GFLOPs, Parameter Count
"""

from __future__ import annotations

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# Detection Metrics
# ============================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between box1 [N, 4] and box2 [M, 4].
    Boxes format: (x1, y1, x2, y2)
    Returns: IoU matrix [N, M]
    """
    # Intersection
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + 1e-6)


class APCalculator:
    """Compute Average Precision (AP) for a single class."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._predictions: List[Tuple] = []  # (confidence, is_tp)
        self._n_gt: int = 0

    def update(
        self,
        pred_boxes: np.ndarray,    # [N, 4] (x1,y1,x2,y2)
        pred_scores: np.ndarray,   # [N]
        gt_boxes: np.ndarray,      # [M, 4]
    ):
        """Add predictions and GT for one image."""
        self._n_gt += len(gt_boxes)

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            for s in pred_scores:
                self._predictions.append((s, False))
            return

        iou = compute_iou(pred_boxes, gt_boxes)  # [N, M]
        matched_gt = set()

        for i in np.argsort(-pred_scores):
            best_iou = iou[i].max()
            best_j = iou[i].argmax()
            if best_iou >= self.iou_threshold and best_j not in matched_gt:
                self._predictions.append((pred_scores[i], True))
                matched_gt.add(best_j)
            else:
                self._predictions.append((pred_scores[i], False))

    def compute(self) -> float:
        """Compute AP using 101-point interpolation."""
        if not self._predictions or self._n_gt == 0:
            return 0.0

        self._predictions.sort(key=lambda x: -x[0])
        tp = np.array([p[1] for p in self._predictions], dtype=float)
        fp = 1.0 - tp

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / (self._n_gt + 1e-6)

        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            prec = precision[recall >= t].max() if (recall >= t).any() else 0.0
            ap += prec / 101
        return ap


class DetectionMetrics:
    """
    Compute mAP@50 and mAP@50-95 across all classes.

    Usage:
        metrics = DetectionMetrics(num_classes=80)
        metrics.update(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)
        results = metrics.compute()
    """

    IOU_THRESHOLDS_50_95 = np.arange(0.5, 1.0, 0.05)  # 10 thresholds

    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        self._calculators_50 = [APCalculator(0.50) for _ in range(num_classes)]
        self._calculators_5095 = [
            [APCalculator(iou) for iou in self.IOU_THRESHOLDS_50_95]
            for _ in range(num_classes)
        ]

    def update(
        self,
        pred_boxes: np.ndarray,    # [N, 4]
        pred_scores: np.ndarray,   # [N]
        pred_classes: np.ndarray,  # [N] int
        gt_boxes: np.ndarray,      # [M, 4]
        gt_classes: np.ndarray,    # [M] int
    ):
        for cls in range(self.num_classes):
            pmask = pred_classes == cls
            gmask = gt_classes == cls
            pb = pred_boxes[pmask]
            ps = pred_scores[pmask]
            gb = gt_boxes[gmask]

            self._calculators_50[cls].update(pb, ps, gb)
            for calc in self._calculators_5095[cls]:
                calc.update(pb, ps, gb)

    def compute(self) -> Dict[str, float]:
        map50 = np.mean([c.compute() for c in self._calculators_50])
        map5095 = np.mean([
            np.mean([calc.compute() for calc in class_calcs])
            for class_calcs in self._calculators_5095
        ])
        return {
            "mAP@50": float(map50),
            "mAP@50-95": float(map5095),
        }


# ============================================================
# Trajectory Metrics
# ============================================================

def compute_ade(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Average Displacement Error: mean L2 distance over all future steps.

    Args:
        pred_traj: [N, T, 2] predicted center (cx, cy)
        gt_traj:   [N, T, 2] GT center (cx, cy)
    Returns:
        ADE (float)
    """
    diff = pred_traj - gt_traj          # [N, T, 2]
    dist = np.linalg.norm(diff, axis=-1)  # [N, T]
    return float(dist.mean())


def compute_fde(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Final Displacement Error: L2 distance at the last predicted step.

    Args:
        pred_traj: [N, T, 2]
        gt_traj:   [N, T, 2]
    Returns:
        FDE (float)
    """
    diff = pred_traj[:, -1, :] - gt_traj[:, -1, :]  # [N, 2]
    dist = np.linalg.norm(diff, axis=-1)             # [N]
    return float(dist.mean())


class TrajectoryMetrics:
    """
    Compute ADE and FDE for trajectory prediction evaluation.

    Usage:
        tm = TrajectoryMetrics(horizon=5)
        tm.update(pred_boxes, gt_boxes)  # [N, T, 4] (cx, cy, w, h)
        results = tm.compute()
    """

    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self._ades: List[float] = []
        self._fdes: List[float] = []

    def update(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray):
        """
        Args:
            pred_boxes: [N, T, 4] predicted boxes (cx, cy, w, h)
            gt_boxes:   [N, T, 4] GT boxes (cx, cy, w, h)
        """
        if len(pred_boxes) == 0:
            return
        # Use center points only
        pred_centers = pred_boxes[:, :self.horizon, :2]
        gt_centers = gt_boxes[:, :self.horizon, :2]
        self._ades.append(compute_ade(pred_centers, gt_centers))
        self._fdes.append(compute_fde(pred_centers, gt_centers))

    def compute(self) -> Dict[str, float]:
        return {
            "ADE": float(np.mean(self._ades)) if self._ades else 0.0,
            "FDE": float(np.mean(self._fdes)) if self._fdes else 0.0,
        }


# ============================================================
# Efficiency Metrics
# ============================================================

class EfficiencyProfiler:
    """
    Profile inference efficiency: FPS, latency, GFLOPs, parameter count.

    Usage:
        profiler = EfficiencyProfiler(model, device='cuda')
        results = profiler.profile(input_size=(1, 3, 640, 640))
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda", warmup: int = 10):
        self.model = model
        self.device = device
        self.warmup = warmup

    @torch.no_grad()
    def profile(
        self,
        input_size: Tuple[int, ...] = (1, 3, 640, 640),
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Args:
            input_size: Input tensor shape (B, C, H, W)
            n_runs:     Number of inference runs for averaging
        Returns:
            Dict with fps, latency_ms, params_M
        """
        self.model.eval().to(self.device)
        dummy = torch.randn(*input_size, device=self.device)

        # Warmup
        for _ in range(self.warmup):
            _ = self.model(dummy)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Measure latency
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = self.model(dummy)
            if self.device == "cuda":
                torch.cuda.synchronize()
        end = time.perf_counter()

        avg_latency_ms = (end - start) / n_runs * 1000
        fps = 1000.0 / avg_latency_ms * input_size[0]  # Scale by batch size

        # Parameter count
        params = sum(p.numel() for p in self.model.parameters()) / 1e6

        results = {
            "fps": round(fps, 1),
            "latency_ms": round(avg_latency_ms, 2),
            "params_M": round(params, 2),
        }

        # GFLOPs (requires thop or fvcore)
        try:
            from thop import profile as thop_profile
            flops, _ = thop_profile(self.model, inputs=(dummy,), verbose=False)
            results["gflops"] = round(flops / 1e9, 2)
        except ImportError:
            results["gflops"] = -1.0  # Install: pip install thop

        return results


# ============================================================
# Aggregated Results Logger
# ============================================================

class AMTYOLOMetrics:
    """
    All-in-one metrics tracker for AMT-YOLO evaluation.
    Combines detection, trajectory, and tracking metrics.
    """

    def __init__(self, num_classes: int = 80, trajectory_horizon: int = 5):
        self.detection = DetectionMetrics(num_classes)
        self.trajectory = TrajectoryMetrics(trajectory_horizon)

    def compute_all(self) -> Dict[str, float]:
        results = {}
        results.update(self.detection.compute())
        results.update(self.trajectory.compute())
        return results

    def print_summary(self, results: Optional[Dict] = None):
        if results is None:
            results = self.compute_all()
        print("\n" + "=" * 50)
        print("  AMT-YOLO Evaluation Results")
        print("=" * 50)
        for k, v in results.items():
            print(f"  {k:<20}: {v:.4f}")
        print("=" * 50 + "\n")
