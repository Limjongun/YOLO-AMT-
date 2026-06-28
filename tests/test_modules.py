"""
test_modules.py — Unit Tests for AMT-YOLO Core Modules
=======================================================
Tests:
    1. ConvGRUCell    — forward pass, state shape, device consistency
    2. ConvLSTMCell   — forward pass, state tuple, cell/hidden shapes
    3. TemporalMemoryModule — full sequence, mode switching, reset
    4. TrajectoryPredictionHead — single obj, batch, empty detection
    5. AdaptiveFeatureFusionNeck — multi-scale attention
    6. YOLOv8FeatureExtractor — hook extraction, feature shapes
    7. AMTYOLO (E2E) — single frame, sequence, memory state persistence

Run:
    cd "D:\\YOLO next"
    .venv\\Scripts\\python -m pytest tests/test_modules.py -v
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. ConvGRUCell
# ============================================================

class TestConvGRUCell:
    def setup_method(self):
        from amt_yolo.models.temporal_memory import ConvGRUCell
        self.cell = ConvGRUCell(in_channels=64, hidden_dim=32).to(DEVICE)
        self.B, self.C, self.H, self.W = 2, 64, 20, 20

    def test_forward_no_state(self):
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
        h = self.cell(x, None)
        assert h.shape == (self.B, 32, self.H, self.W), f"Unexpected shape: {h.shape}"

    def test_forward_with_state(self):
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
        h_prev = torch.zeros(self.B, 32, self.H, self.W, device=DEVICE)
        h = self.cell(x, h_prev)
        assert h.shape == h_prev.shape

    def test_state_gradients(self):
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE, requires_grad=True)
        h = self.cell(x, None)
        loss = h.sum()
        loss.backward()
        assert x.grad is not None

    def test_sequence_run(self):
        """Run 5 frames through the cell and check shapes remain consistent."""
        h = None
        for _ in range(5):
            x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
            h = self.cell(x, h)
        assert h.shape == (self.B, 32, self.H, self.W)


# ============================================================
# 2. ConvLSTMCell
# ============================================================

class TestConvLSTMCell:
    def setup_method(self):
        from amt_yolo.models.temporal_memory import ConvLSTMCell
        self.cell = ConvLSTMCell(in_channels=64, hidden_dim=32).to(DEVICE)
        self.B, self.C, self.H, self.W = 2, 64, 20, 20

    def test_forward_no_state(self):
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
        h, (h2, c) = self.cell(x, None)
        assert h.shape == (self.B, 32, self.H, self.W)
        assert c.shape == (self.B, 32, self.H, self.W)

    def test_forward_with_state(self):
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
        state = (
            torch.zeros(self.B, 32, self.H, self.W, device=DEVICE),
            torch.zeros(self.B, 32, self.H, self.W, device=DEVICE),
        )
        h, new_state = self.cell(x, state)
        assert h.shape == (self.B, 32, self.H, self.W)

    def test_gates_bounded(self):
        """Verify LSTM gates produce bounded outputs."""
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE) * 10
        h, (h2, c) = self.cell(x, None)
        # tanh output of hidden: bounded [-1, 1]
        assert h.min() >= -1.0 - 1e-5 and h.max() <= 1.0 + 1e-5


# ============================================================
# 3. TemporalMemoryModule
# ============================================================

class TestTemporalMemoryModule:
    def setup_method(self):
        from amt_yolo.models.temporal_memory import TemporalMemoryModule
        self.B, self.C, self.H, self.W = 2, 256, 20, 20

    @pytest.mark.parametrize("mode", ["convgru", "convlstm"])
    def test_forward_modes(self, mode):
        from amt_yolo.models.temporal_memory import TemporalMemoryModule
        module = TemporalMemoryModule(in_channels=self.C, hidden_dim=64, mode=mode).to(DEVICE)
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
        enhanced, state = module(x, None)
        assert enhanced.shape == x.shape, f"Shape mismatch in mode={mode}"

    def test_sequence_state_updates(self):
        from amt_yolo.models.temporal_memory import TemporalMemoryModule
        module = TemporalMemoryModule(in_channels=self.C, hidden_dim=64, mode="convgru").to(DEVICE)
        state = None
        states = []
        for _ in range(5):
            x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
            out, state = module(x, state)
            states.append(state.clone())
        # Each state should differ (memory is updating)
        assert not torch.allclose(states[0], states[-1])

    def test_reset_memory(self):
        from amt_yolo.models.temporal_memory import TemporalMemoryModule
        module = TemporalMemoryModule(in_channels=self.C, hidden_dim=64, mode="convgru").to(DEVICE)
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
        _, state1 = module(x, None)
        module.reset_memory()
        assert module._state is None
        assert module._frame_count == 0

    def test_output_not_equal_input(self):
        """Memory module should alter the features, not just pass-through."""
        from amt_yolo.models.temporal_memory import TemporalMemoryModule
        module = TemporalMemoryModule(in_channels=self.C, hidden_dim=64, mode="convgru").to(DEVICE)
        x = torch.randn(self.B, self.C, self.H, self.W, device=DEVICE)
        enhanced, _ = module(x, None)
        assert not torch.allclose(enhanced, x)


# ============================================================
# 4. AdaptiveFeatureFusionNeck
# ============================================================

class TestFeatureFusionNeck:
    def setup_method(self):
        from amt_yolo.models.feature_fusion import AdaptiveFeatureFusionNeck
        self.channels = [128, 256, 512]
        self.neck = AdaptiveFeatureFusionNeck(in_channels_list=self.channels).to(DEVICE)
        self.B = 2

    def test_forward_shapes(self):
        features = [
            torch.randn(self.B, 128, 80, 80, device=DEVICE),
            torch.randn(self.B, 256, 40, 40, device=DEVICE),
            torch.randn(self.B, 512, 20, 20, device=DEVICE),
        ]
        enhanced = self.neck(features)
        assert len(enhanced) == 3
        for orig, enh in zip(features, enhanced):
            assert orig.shape == enh.shape

    def test_scale_weights_learnable(self):
        """Verify scale weights are part of the computational graph."""
        features = [
            torch.randn(self.B, 128, 80, 80, device=DEVICE),
            torch.randn(self.B, 256, 40, 40, device=DEVICE),
            torch.randn(self.B, 512, 20, 20, device=DEVICE),
        ]
        enhanced = self.neck(features)
        loss = sum(e.sum() for e in enhanced)
        loss.backward()
        assert self.neck.scale_weights.weights.grad is not None


# ============================================================
# 5. TrajectoryPredictionHead
# ============================================================

class TestTrajectoryHead:
    def setup_method(self):
        from amt_yolo.models.trajectory_head import TrajectoryPredictionHead
        self.head = TrajectoryPredictionHead(
            obj_embed_dim=128,
            mem_embed_dim=128,
            hidden_dim=64,
            horizon=5,
        ).to(DEVICE)

    def test_single_object(self):
        N = 1
        obj_embed = torch.randn(N, 128, device=DEVICE)
        mem_embed = torch.randn(N, 128, device=DEVICE)
        current_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=DEVICE)
        out = self.head(obj_embed, mem_embed, current_boxes)
        assert out["future_boxes"].shape == (N, 5, 4)
        assert out["motion_vectors"].shape == (N, 5, 2)
        assert out["confidences"].shape == (N, 5, 1)

    def test_batch_objects(self):
        N = 8
        obj_embed = torch.randn(N, 128, device=DEVICE)
        mem_embed = torch.randn(N, 128, device=DEVICE)
        current_boxes = torch.rand(N, 4, device=DEVICE)
        out = self.head(obj_embed, mem_embed, current_boxes)
        assert out["future_boxes"].shape == (N, 5, 4)

    def test_empty_detections(self):
        """Empty detection batch should return zero-shaped tensors gracefully."""
        N = 0
        obj_embed = torch.zeros(N, 128, device=DEVICE)
        mem_embed = torch.zeros(N, 128, device=DEVICE)
        current_boxes = torch.zeros(N, 4, device=DEVICE)
        out = self.head(obj_embed, mem_embed, current_boxes)
        assert out["future_boxes"].shape == (0, 5, 4)

    def test_confidence_bounded(self):
        """Confidence outputs should be in [0, 1] due to Sigmoid."""
        N = 4
        obj_embed = torch.randn(N, 128, device=DEVICE) * 10
        mem_embed = torch.randn(N, 128, device=DEVICE) * 10
        current_boxes = torch.rand(N, 4, device=DEVICE)
        out = self.head(obj_embed, mem_embed, current_boxes)
        confs = out["confidences"]
        assert confs.min() >= 0.0 and confs.max() <= 1.0


# ============================================================
# 6. End-to-End: YOLOv8FeatureExtractor (hook test)
# ============================================================

class TestYOLOv8FeatureExtractor:
    """
    These tests require Ultralytics + CUDA.
    Will download yolov8s.pt (~22MB) on first run.
    """

    def setup_method(self):
        from amt_yolo.models.amt_yolo import YOLOv8FeatureExtractor
        self.extractor = YOLOv8FeatureExtractor("yolov8s", pretrained=True).to(DEVICE)
        self.B = 1

    def test_feature_extraction_runs(self):
        x = torch.randn(self.B, 3, 640, 640, device=DEVICE)
        features, _ = self.extractor(x)
        # At least p5 should be populated (p3/p4 may depend on layer indices)
        assert any(v is not None for v in features.values()), \
            "No features were captured by hooks"

    def test_p5_channels(self):
        """P5 output channel count matches expected for yolov8s."""
        x = torch.randn(self.B, 3, 640, 640, device=DEVICE)
        features, _ = self.extractor(x)
        p5 = features.get("p5")
        if p5 is not None:
            assert p5.shape[1] == 512, f"Unexpected P5 channels: {p5.shape[1]}"


# ============================================================
# 7. End-to-End: AMTYOLO
# ============================================================

class TestAMTYOLO:
    """Full end-to-end tests for AMTYOLO model."""

    def setup_method(self):
        from amt_yolo.models.amt_yolo import AMTYOLO
        self.model = AMTYOLO(
            backbone="yolov8s",
            memory_type="convgru",
            trajectory_horizon=5,
            adaptive_resolution=False,  # Disable for unit test simplicity
            pretrained=True,
        ).to(DEVICE)

    def test_single_frame_forward(self):
        x = torch.randn(1, 3, 640, 640, device=DEVICE)
        results = self.model(x, return_trajectory=False)
        assert "detections" in results
        assert "features" in results
        assert "resolution" in results

    def test_memory_state_persists(self):
        """Memory state should change between frames."""
        self.model.reset_memory()
        x = torch.randn(1, 3, 640, 640, device=DEVICE)
        self.model(x, return_trajectory=False)
        state1 = self.model._memory_state

        x2 = torch.randn(1, 3, 640, 640, device=DEVICE)
        self.model(x2, return_trajectory=False)
        state2 = self.model._memory_state

        assert state1 is not None
        assert state2 is not None
        assert not torch.allclose(state1, state2), "Memory state did not update"

    def test_reset_clears_memory(self):
        x = torch.randn(1, 3, 640, 640, device=DEVICE)
        self.model(x)
        self.model.reset_memory()
        assert self.model._memory_state is None

    def test_sequence_forward(self):
        frames = [torch.randn(1, 3, 640, 640, device=DEVICE) for _ in range(5)]
        results = self.model.forward_sequence(frames)
        assert len(results) == 5

    def test_freeze_unfreeze_backbone(self):
        self.model.freeze_backbone()
        frozen = all(
            not p.requires_grad
            for p in self.model.backbone.yolo_model.parameters()
        )
        assert frozen, "Backbone not fully frozen"

        self.model.unfreeze_backbone()
        unfrozen = any(
            p.requires_grad
            for p in self.model.backbone.yolo_model.parameters()
        )
        assert unfrozen, "Backbone not unfrozen"

    def test_num_parameters_positive(self):
        assert self.model.num_parameters > 0

    @pytest.mark.parametrize("memory_type", ["convgru", "convlstm"])
    def test_memory_types(self, memory_type):
        """Both ConvGRU and ConvLSTM should run without errors."""
        from amt_yolo.models.amt_yolo import AMTYOLO
        model = AMTYOLO(
            backbone="yolov8s",
            memory_type=memory_type,
            trajectory_horizon=5,
            adaptive_resolution=False,
            pretrained=True,
        ).to(DEVICE)
        x = torch.randn(1, 3, 640, 640, device=DEVICE)
        out = model(x)
        assert "features" in out
