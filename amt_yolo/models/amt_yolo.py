"""
amt_yolo.py — AMT-YOLO Main Model
===================================
End-to-end Adaptive Memory Trajectory YOLO.
Integrates:
    1. AdaptiveResolutionRouter  — scene-adaptive input resolution
    2. YOLOv8 Backbone           — feature extraction (P3, P4, P5 via hook)
    3. AdaptiveFeatureFusionNeck — attention-enhanced multi-scale fusion
    4. TemporalMemoryModule      — inter-frame memory (ConvGRU / ConvLSTM)
    5. Detection Head            — boxes, classes (via Ultralytics)
    6. TrajectoryPredictionHead  — future position prediction (5-10 frames)

Base: YOLOv8 (Ultralytics)
Hardware: Optimized for RTX 4050 6GB VRAM
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from amt_yolo.models.adaptive_resolution import AdaptiveResolutionRouter
from amt_yolo.models.temporal_memory import TemporalMemoryModule
from amt_yolo.models.trajectory_head import TrajectoryPredictionHead
from amt_yolo.models.feature_fusion import AdaptiveFeatureFusionNeck


# ============================================================
# YOLOv8 Feature Hook Extractor
# ============================================================

class YOLOv8FeatureExtractor(nn.Module):
    """
    Wraps Ultralytics YOLOv8 model and extracts intermediate feature maps
    from specific backbone layers using PyTorch forward hooks.

    YOLOv8 backbone layer indices (approximate, varies by variant):
        - P3 (stride 8):  layer index 4  — small object features
        - P4 (stride 16): layer index 6  — medium object features
        - P5 (stride 32): layer index 9  — large object features, richest semantics

    We hook P5 (highest-level semantics) as primary input to TemporalMemoryModule.
    P3/P4/P5 are also extracted for AdaptiveFeatureFusionNeck.
    """

    # Layer indices within the Ultralytics DetectionModel's model Sequential
    LAYER_INDICES = {
        "yolov8n": {"p3": 4,  "p4": 6,  "p5": 9},
        "yolov8s": {"p3": 4,  "p4": 6,  "p5": 9},
        "yolov8m": {"p3": 4,  "p4": 6,  "p5": 9},
        "yolov8l": {"p3": 4,  "p4": 6,  "p5": 9},
    }

    # Output channels per backbone variant (before neck)
    OUTPUT_CHANNELS = {
        "yolov8n": {"p3": 64,  "p4": 128, "p5": 256},
        "yolov8s": {"p3": 128, "p4": 256, "p5": 512},
        "yolov8m": {"p3": 192, "p4": 384, "p5": 576},
        "yolov8l": {"p3": 256, "p4": 512, "p5": 512},
    }

    def __init__(self, backbone_name: str = "yolov8s", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self._features: Dict[str, Optional[torch.Tensor]] = {
            "p3": None, "p4": None, "p5": None
        }
        self._hooks: List = []

        try:
            from ultralytics import YOLO as UltralyticsYOLO
            model_file = f"{backbone_name}.pt" if pretrained else f"{backbone_name}.yaml"
            # Do NOT store yolo in self._yolo, because YOLO overrides .train() and crashes PyTorch recursive train()
            yolo_wrapper = UltralyticsYOLO(model_file)
            self.yolo_model = yolo_wrapper.model  # Ultralytics DetectionModel
            self.yolo_model.eval()
        except ImportError:
            raise ImportError("Ultralytics required. Run: pip install ultralytics")

        # Register hooks on backbone layers
        self._register_hooks()

        # Expose output channel counts
        self.channels = self.OUTPUT_CHANNELS.get(
            backbone_name, {"p3": 128, "p4": 256, "p5": 512}
        )

    def _register_hooks(self):
        """Register forward hooks on P3, P4, P5 backbone layers."""
        self._remove_hooks()
        indices = self.LAYER_INDICES.get(self.backbone_name, {"p3": 4, "p4": 6, "p5": 9})
        layers = list(self.yolo_model.model.children())

        for scale, idx in indices.items():
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(
                    self._make_hook(scale)
                )
                self._hooks.append(hook)

    def _make_hook(self, scale: str):
        """Factory that creates a closure hook for a given scale name."""
        def hook(module, input, output):
            self._features[scale] = output
        return hook

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Run YOLOv8 forward pass and collect hooked features.

        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            features: Dict {'p3': tensor, 'p4': tensor, 'p5': tensor}
            yolo_out: Raw Ultralytics model output (detection results)
        """
        # Clear previous hook outputs
        self._features = {"p3": None, "p4": None, "p5": None}

        # Forward pass — hooks fire automatically during this
        with torch.no_grad() if not self.training else torch.enable_grad():
            yolo_out = self.yolo_model(x)

        return dict(self._features), yolo_out

    def __del__(self):
        self._remove_hooks()


# ============================================================
# AMT-YOLO Main Model
# ============================================================

class AMTYOLO(nn.Module):
    """
    AMT-YOLO: Adaptive Memory Trajectory YOLO

    End-to-end model integrating adaptive resolution, temporal memory,
    and trajectory prediction on top of YOLOv8.

    Args:
        backbone:             YOLOv8 variant — 'yolov8n' | 'yolov8s' | 'yolov8m'
        memory_type:          'convgru' | 'convlstm' | 'none'
        trajectory_horizon:   Number of future frames to predict (5–10)
        adaptive_resolution:  Whether to use Adaptive Resolution Router
        resolutions:          Candidate resolutions for adaptive routing
        pretrained:           Load YOLOv8 COCO pretrained weights
        num_classes:          Number of detection classes (80 for COCO)
        obj_embed_dim:        Tracking embedding dimension
        memory_hidden_dim:    Memory hidden channels (128 for 6GB VRAM)
        use_feature_fusion:   Use AdaptiveFeatureFusionNeck on P3/P4/P5
    """

    def __init__(
        self,
        backbone: str = "yolov8s",
        memory_type: str = "convgru",
        trajectory_horizon: int = 5,
        adaptive_resolution: bool = True,
        resolutions: List[int] = [640, 768],
        pretrained: bool = True,
        num_classes: int = 80,
        obj_embed_dim: int = 128,
        memory_hidden_dim: int = 128,
        use_feature_fusion: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.memory_type = memory_type
        self.trajectory_horizon = trajectory_horizon
        self.use_adaptive_resolution = adaptive_resolution
        self.num_classes = num_classes

        # --- 1. Adaptive Resolution Router ---
        if adaptive_resolution:
            self.resolution_router = AdaptiveResolutionRouter(
                resolutions=resolutions,
                thresholds=[0.35],
                estimator="edge",
                training_max_res=max(resolutions),
            )
        else:
            self.resolution_router = None

        # --- 2. YOLOv8 Backbone with P3/P4/P5 feature hooks ---
        self.backbone = YOLOv8FeatureExtractor(backbone, pretrained)
        channels = self.backbone.channels  # {'p3': C3, 'p4': C4, 'p5': C5}
        p5_ch = channels["p5"]

        # --- 3. Adaptive Feature Fusion Neck ---
        if use_feature_fusion:
            self.fusion_neck = AdaptiveFeatureFusionNeck(
                in_channels_list=[channels["p3"], channels["p4"], channels["p5"]],
                use_se_attention=True,
                use_spatial_attention=True,
                use_adaptive_weights=True,
            )
        else:
            self.fusion_neck = None

        # --- 4. Temporal Memory Module (operates on P5) ---
        if memory_type != "none":
            self.temporal_memory = TemporalMemoryModule(
                in_channels=p5_ch,
                hidden_dim=memory_hidden_dim,
                mode=memory_type,
                kernel_size=3,
                reset_every_n=0,
            )
        else:
            self.temporal_memory = None

        # --- 5. Tracking Embedding Head (from memory-enhanced P5) ---
        self.tracking_embed = nn.Sequential(
            nn.Conv2d(p5_ch, obj_embed_dim, 1, bias=False),
            nn.BatchNorm2d(obj_embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pool → [B, embed_dim, 1, 1]
        )

        # --- 6. Trajectory Prediction Head ---
        if trajectory_horizon > 0:
            self.trajectory_head = TrajectoryPredictionHead(
                obj_embed_dim=obj_embed_dim,
                mem_embed_dim=memory_hidden_dim if memory_type != "none" else obj_embed_dim,
                hidden_dim=128,
                horizon=trajectory_horizon,
                teacher_forcing=0.5,
            )
        else:
            self.trajectory_head = None

        # Memory state (persists across frames in a sequence)
        self._memory_state = None

    def reset_memory(self):
        """Reset temporal memory. Call between video sequences."""
        if self.temporal_memory is not None:
            self.temporal_memory.reset_memory()
        self._memory_state = None

    def forward(
        self,
        x: torch.Tensor,
        current_boxes: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
        teacher_boxes: Optional[torch.Tensor] = None,
        return_trajectory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a single frame.

        Args:
            x:                  Input frame [B, 3, H, W], normalized [0, 1]
            current_boxes:      Current detection boxes [N, 4] (cx, cy, w, h)
            teacher_boxes:      GT future boxes [N, horizon, 4] (training only)
            return_trajectory:  Whether to run trajectory prediction

        Returns:
            dict with keys:
                'detections':    Raw YOLO output
                'p3'/'p4'/'p5': Backbone feature maps (enhanced)
                'features':      Memory-enhanced P5 [B, C, H', W']
                'embeddings':    Tracking embeddings [B, embed_dim]
                'trajectory':    Future position predictions (if enabled)
                'resolution':    Selected resolution (int)
                'complexity':    Scene complexity score (float)
                'memory_state':  Updated memory state
        """
        results = {}

        # ── Step 1: Adaptive Resolution ──────────────────────────────────
        if self.resolution_router is not None:
            x_in, resolution, complexity = self.resolution_router(x)
            results["resolution"] = resolution
            results["complexity"] = complexity
        else:
            x_in = x
            results["resolution"] = x.shape[-1]
            results["complexity"] = 0.0

        # ── Step 2-4: Feature Extraction + Fusion + Memory ───────────────
        # Single autocast scope for the entire processing chain.
        # Explicit .float() cast after backbone prevents FP16/FP32 mismatch
        # in downstream Linear and BatchNorm layers.
        amp_enabled = x_in.is_cuda
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            backbone_features, yolo_out = self.backbone(x_in)

        results["detections"] = yolo_out

        # Cast backbone outputs to float32 — required for Linear/BN compatibility
        p3 = backbone_features.get("p3")
        p4 = backbone_features.get("p4")
        p5 = backbone_features.get("p5")
        if p3 is not None: p3 = p3.float()
        if p4 is not None: p4 = p4.float()
        if p5 is not None: p5 = p5.float()

        # ── Step 3: Adaptive Feature Fusion Neck ─────────────────────────
        if self.fusion_neck is not None and all(f is not None for f in [p3, p4, p5]):
            fused = self.fusion_neck([p3, p4, p5])
            p3, p4, p5 = fused[0], fused[1], fused[2]

        results["p3"] = p3
        results["p4"] = p4
        results["p5"] = p5

        # ── Step 4: Temporal Memory (on P5) ──────────────────────────────
        if self.temporal_memory is not None and p5 is not None:
            p5_enhanced, self._memory_state = self.temporal_memory(
                p5, self._memory_state
            )
        else:
            p5_enhanced = p5

        results["features"] = p5_enhanced
        results["memory_state"] = self._memory_state

        # ── Step 5: Tracking Embeddings ───────────────────────────────────
        if p5_enhanced is not None:
            embed = self.tracking_embed(p5_enhanced)  # [B, embed_dim, 1, 1]
            embed = embed.view(embed.shape[0], -1)    # [B, embed_dim]
            results["embeddings"] = embed
        else:
            results["embeddings"] = None

        # ── Step 6: Trajectory Prediction ────────────────────────────────
        if (
            return_trajectory
            and self.trajectory_head is not None
            and current_boxes is not None
            and results.get("embeddings") is not None
            and batch_idx is not None
        ):
            embed = results["embeddings"]
            
            # Map [B, 256] image embeddings to [N, 256] object embeddings
            # batch_idx is [N] containing the image index in the batch for each object
            obj_embed = embed[batch_idx]
            
            # Pool memory feature per object
            mem_feat = obj_embed  # Simplified: use tracking embed as memory proxy
            traj_out = self.trajectory_head(
                obj_embed=obj_embed,
                mem_embed=mem_feat,
                current_boxes=current_boxes,
                teacher_boxes=teacher_boxes,
            )
            results["trajectory"] = traj_out
        else:
            results["trajectory"] = None

        return results

    def forward_sequence(
        self,
        frames: List[torch.Tensor],
        reset_between: bool = False,
    ) -> List[Dict]:
        """
        Process a sequence of frames while maintaining memory continuity.
        Used during training (temporal sequence batches) and inference.

        Args:
            frames:         List of frame tensors [B, 3, H, W]
            reset_between:  If True, reset memory between each frame (ablation)
        Returns:
            List of result dicts per frame
        """
        all_results = []
        self.reset_memory()

        for frame in frames:
            result = self.forward(frame, return_trajectory=True)
            all_results.append(result)
            if reset_between:
                self.reset_memory()

        return all_results

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count (excluding frozen YOLOv8 backbone)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Freeze YOLOv8 backbone weights during Stage 1 training."""
        for param in self.backbone.yolo_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for Stage 2 fine-tuning."""
        for param in self.backbone.yolo_model.parameters():
            param.requires_grad = True

    def __repr__(self) -> str:
        return (
            f"AMTYOLO(\n"
            f"  backbone={self.backbone_name},\n"
            f"  memory_type={self.memory_type},\n"
            f"  trajectory_horizon={self.trajectory_horizon},\n"
            f"  adaptive_resolution={self.use_adaptive_resolution},\n"
            f"  num_classes={self.num_classes},\n"
            f"  trainable_params={self.num_parameters:,}\n"
            f")"
        )
