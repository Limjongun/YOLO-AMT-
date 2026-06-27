"""
verify_setup.py — AMT-YOLO Environment Verification Script
============================================================
Run this script after installing requirements to verify:
  1. Python version
  2. CUDA availability and GPU info
  3. PyTorch version
  4. Ultralytics YOLO
  5. OpenCV
  6. Package import check

Usage:
    python verify_setup.py
"""

import sys
import platform


def check_python():
    version = sys.version_info
    ok = version >= (3, 12)
    status = "✅" if ok else "❌"
    print(f"{status} Python {version.major}.{version.minor}.{version.micro} {'(OK)' if ok else '(Need 3.12+)'}")
    return ok


def check_torch():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_status = "✅" if cuda_available else "⚠️ "
        print(f"✅ PyTorch {torch.__version__}")
        print(f"{cuda_status} CUDA Available: {cuda_available}")
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {vram:.1f} GB")
            if vram < 6.0:
                print(f"   ⚠️  WARNING: Less than 6GB VRAM — may need to reduce batch size further")
            elif vram < 8.0:
                print(f"   ⚠️  RTX 4050 6GB detected — using AMP + gradient checkpointing config")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        return True
    except ImportError:
        print("❌ PyTorch NOT installed — run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_ultralytics():
    try:
        import ultralytics
        print(f"✅ Ultralytics {ultralytics.__version__}")
        # Test YOLOv8 model load
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print(f"   YOLOv8n loaded successfully (params: {sum(p.numel() for p in model.model.parameters()):,})")
        return True
    except ImportError:
        print("❌ Ultralytics NOT installed — run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"⚠️  Ultralytics installed but error loading model: {e}")
        return False


def check_opencv():
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV NOT installed")
        return False


def check_numpy():
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        return True
    except ImportError:
        print("❌ NumPy NOT installed")
        return False


def check_wandb():
    try:
        import wandb
        print(f"✅ WandB {wandb.__version__}")
        return True
    except ImportError:
        print("⚠️  WandB not installed (optional) — run: pip install wandb")
        return False


def check_amt_yolo_package():
    try:
        import amt_yolo
        print(f"✅ amt_yolo package importable (v{amt_yolo.__version__})")
        return True
    except ImportError as e:
        print(f"⚠️  amt_yolo package not yet installed — run: pip install -e .")
        return False


def check_amp():
    """Verify AMP (Automatic Mixed Precision) works on the GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  AMP check skipped (no CUDA)")
            return False
        # Quick AMP test
        with torch.cuda.amp.autocast():
            a = torch.randn(10, 10, device='cuda')
            b = torch.randn(10, 10, device='cuda')
            c = a @ b
        print(f"✅ AMP (Automatic Mixed Precision) — FP16 OK")
        return True
    except Exception as e:
        print(f"❌ AMP test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  AMT-YOLO Environment Verification")
    print(f"  OS: {platform.system()} {platform.release()}")
    print("=" * 60)
    print()

    results = {
        "Python": check_python(),
        "PyTorch+CUDA": check_torch(),
        "Ultralytics": check_ultralytics(),
        "OpenCV": check_opencv(),
        "NumPy": check_numpy(),
        "WandB": check_wandb(),
        "AMT-YOLO pkg": check_amt_yolo_package(),
        "AMP": check_amp(),
    }

    print()
    print("=" * 60)
    passed = sum(v for v in results.values())
    total = len(results)
    print(f"  Results: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉 Environment is ready for AMT-YOLO development!")
    elif passed >= total - 2:
        print("  ⚠️  Almost ready — install missing optional packages")
    else:
        print("  ❌ Please install missing required packages first")
    print("=" * 60)


if __name__ == "__main__":
    main()
