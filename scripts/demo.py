"""
demo.py — AMT-YOLO Real-Time Demo
====================================
Runs AMT-YOLO inference on webcam or video file.
Visualizes: detection boxes + tracking IDs + trajectory overlays.

Usage:
    # Webcam demo
    python scripts/demo.py --source 0

    # Video file
    python scripts/demo.py --source video.mp4 --show-trajectory

    # Benchmark FPS only
    python scripts/demo.py --source 0 --benchmark
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
import numpy as np
from rich.console import Console

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="AMT-YOLO Demo")
    parser.add_argument("--source", default="0", help="Video source: 0 (webcam) or path to video")
    parser.add_argument("--config", default="configs/amt_yolo_small.yaml")
    parser.add_argument("--weights", default=None, help="Path to model weights (.pt)")
    parser.add_argument("--show-trajectory", action="store_true", help="Overlay trajectory predictions")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark FPS only")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--save", type=str, default=None, help="Save output to video file")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def draw_trajectory(
    frame: np.ndarray,
    future_boxes: np.ndarray,  # [T, 4] (cx, cy, w, h) normalized
    color: tuple = (0, 255, 100),
    alpha: float = 0.6,
) -> np.ndarray:
    """Draw predicted trajectory on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    centers = []
    for i, box in enumerate(future_boxes):
        cx, cy = int(box[0] * w), int(box[1] * h)
        centers.append((cx, cy))
        radius = max(3, 8 - i)  # Shrinking dots for further predictions
        opacity = max(0.2, 1.0 - i * 0.1)
        cv2.circle(overlay, (cx, cy), radius, color, -1)

    # Draw connecting lines
    for i in range(1, len(centers)):
        cv2.line(overlay, centers[i-1], centers[i], color, 2)

    return cv2.addWeighted(frame, 1 - alpha * 0.3, overlay, alpha * 0.3, 0)


def run_demo(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[cyan]AMT-YOLO Demo[/cyan]")
    console.print(f"  Source: {args.source}")
    console.print(f"  Device: {device}")
    console.print(f"  Show trajectory: {args.show_trajectory}")

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        console.print(f"[red]❌ Cannot open source: {args.source}[/red]")
        return

    # Initialize model
    try:
        from amt_yolo.models.amt_yolo import AMTYOLO
        model = AMTYOLO(
            backbone="yolov8n",
            memory_type="convgru",
            trajectory_horizon=5,
            adaptive_resolution=True,
        ).to(device)
        model.eval()
        if args.weights:
            ckpt = torch.load(args.weights, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            console.print(f"[green]✅ Weights loaded: {args.weights}[/green]")
        else:
            console.print("[yellow]⚠ No weights loaded — using untrained model for demo structure[/yellow]")
    except Exception as e:
        console.print(f"[red]Model error: {e}[/red]")
        return

    # Video writer (optional)
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, 30, (w, h))

    fps_history = []
    frame_count = 0
    model.reset_memory()

    console.print("[green]▶ Running... Press 'q' to quit[/green]")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        frame_count += 1

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        # Inference Step 1: Detect objects and extract memory/embeddings
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                results = model(tensor, return_trajectory=False)
                
                # Apply NMS (Ultralytics standard output)
                from ultralytics.utils.nms import non_max_suppression
                preds = non_max_suppression(results["detections"], conf_thres=args.conf, iou_thres=0.45)
                
                det = preds[0] # batch size 1
                
                # Inference Step 2: Predict trajectories for detected objects
                if args.show_trajectory and len(det) > 0 and model.trajectory_head is not None:
                    # det format: [x1, y1, x2, y2, conf, cls] at internal resolution
                    boxes = det[:, :4].clone()
                    h_img, w_img = frame.shape[:2]
                    resolution = results.get("resolution", 640)
                    
                    # Scale boxes back to original image size
                    ratio_x = w_img / resolution
                    ratio_y = h_img / resolution
                    boxes[:, [0, 2]] *= ratio_x
                    boxes[:, [1, 3]] *= ratio_y
                    
                    # Update det with scaled boxes for drawing later
                    det[:, :4] = boxes
                    
                    # normalize for trajectory head
                    boxes[:, [0, 2]] /= w_img
                    boxes[:, [1, 3]] /= h_img
                    
                    # convert to cx, cy, w, h
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2
                    w = boxes[:, 2] - boxes[:, 0]
                    h = boxes[:, 3] - boxes[:, 1]
                    current_boxes = torch.stack([cx, cy, w, h], dim=1)
                    
                    # Query Trajectory Head
                    batch_idx = torch.zeros(len(current_boxes), dtype=torch.long, device=device)
                    obj_embed = results["embeddings"][batch_idx]
                    
                    traj_out = model.trajectory_head(
                        obj_embed=obj_embed,
                        mem_embed=obj_embed,
                        current_boxes=current_boxes
                    )
                    
                    future_boxes = traj_out["future_boxes"].cpu().numpy() # [N, Horizon, 4]
                    confidences = traj_out["confidences"].cpu().numpy()
                else:
                    future_boxes = None

        t1 = time.perf_counter()
        fps = 1.0 / (t1 - t0)
        fps_history.append(fps)

        # Draw Bounding Boxes
        if len(det) > 0:
            for i, d in enumerate(det):
                x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw trajectory if available
                if future_boxes is not None:
                    traj_boxes = future_boxes[i]
                    frame = draw_trajectory(frame, traj_boxes, color=(0, 255, 255))

        # Draw FPS + resolution info
        avg_fps = sum(fps_history[-30:]) / min(len(fps_history), 30)
        resolution = results.get("resolution", 640)
        complexity = results.get("complexity", 0.0)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Res: {resolution}px | Complexity: {complexity:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"Memory: convgru | Horizon: 5f",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if writer:
            writer.write(frame)

        if not args.benchmark:
            cv2.imshow("AMT-YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.benchmark and frame_count >= 100:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final FPS report
    if fps_history:
        console.print(f"\n[bold]Benchmark Results:[/bold]")
        console.print(f"  Average FPS: {sum(fps_history)/len(fps_history):.1f}")
        console.print(f"  Min FPS:     {min(fps_history):.1f}")
        console.print(f"  Max FPS:     {max(fps_history):.1f}")
        console.print(f"  Frames:      {frame_count}")


def main():
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
