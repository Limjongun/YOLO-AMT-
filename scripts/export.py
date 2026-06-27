"""
export.py — AMT-YOLO Model Export (ONNX + TensorRT)

Usage:
    # Export to ONNX
    python scripts/export.py --weights experiments/checkpoints/final.pt --format onnx

    # Export to TensorRT
    python scripts/export.py --weights experiments/checkpoints/final.pt --format tensorrt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console

console = Console()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--format", choices=["onnx", "tensorrt"], default="onnx")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch axis")
    return parser.parse_args()


def export_onnx(model, dummy_input, output_path: str, dynamic: bool = False):
    """Export model to ONNX format."""
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    console.print(f"[green]✅ ONNX exported:[/green] {output_path}")
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    console.print("[green]✅ ONNX model verified[/green]")


def export_tensorrt(onnx_path: str, output_path: str, input_size: int):
    """Export ONNX to TensorRT (requires tensorrt installed)."""
    try:
        import tensorrt as trt
        console.print(f"[green]TensorRT {trt.__version__} found[/green]")
        # TensorRT export implementation — see Phase 6
        console.print("[yellow]TensorRT export: Full implementation in Phase 6[/yellow]")
    except ImportError:
        console.print("[red]TensorRT not installed. Install NVIDIA TensorRT first.[/red]")
        console.print("  https://developer.nvidia.com/tensorrt")


def main():
    args = parse_args()

    from amt_yolo.models.amt_yolo import AMTYOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AMTYOLO(backbone="yolov8n", memory_type="convgru", trajectory_horizon=5)
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval().to(device)
    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)

    if args.format == "onnx":
        out_path = args.output or args.weights.replace(".pt", ".onnx")
        export_onnx(model, dummy, out_path, args.dynamic)
    elif args.format == "tensorrt":
        onnx_path = args.weights.replace(".pt", ".onnx")
        out_path = args.output or args.weights.replace(".pt", ".trt")
        export_onnx(model, dummy, onnx_path, args.dynamic)
        export_tensorrt(onnx_path, out_path, args.input_size)


if __name__ == "__main__":
    main()
