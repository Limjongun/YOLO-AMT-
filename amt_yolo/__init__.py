"""
AMT-YOLO: Adaptive Memory Trajectory YOLO
==========================================

A novel YOLO-based architecture for real-time video object detection,
multi-object tracking, and trajectory prediction.

Three core innovations:
    1. Adaptive Resolution Router      - scene-adaptive input resolution
    2. Temporal Memory Module          - ConvGRU / ConvLSTM inter-frame memory
    3. Trajectory Prediction Head      - 5-10 frame future prediction

Base Model: YOLOv8 (Ultralytics)
Target Hardware: RTX 4050 6GB VRAM (optimized with AMP + gradient accumulation)
Research Target: CVPR
"""

__version__ = "0.1.0"
__author__ = "AMT-YOLO Research Team"

# Package-level imports (lazy to avoid import errors before install)
try:
    from amt_yolo.models.amt_yolo import AMTYOLO
    __all__ = ["AMTYOLO"]
except ImportError:
    # Dependencies may not be installed yet
    __all__ = []
