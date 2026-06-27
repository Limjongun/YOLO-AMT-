"""
evaluate.py — AMT-YOLO Evaluation & Ablation Study Script

Usage:
    # Evaluate single config
    python scripts/evaluate.py --config configs/amt_yolo_base.yaml --dataset mot17

    # Run full ablation study
    python scripts/evaluate.py --ablation --output experiments/results/ablation.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()

ABLATION_CONFIGS = [
    ("Baseline YOLOv8",        "configs/ablation/fixed_resolution.yaml",  "configs/ablation/no_memory.yaml"),
    ("+ ConvGRU Memory",       "configs/ablation/convgru_only.yaml",       None),
    ("+ ConvLSTM Memory",      "configs/ablation/convlstm_only.yaml",      None),
    ("+ Trajectory Only",      "configs/ablation/no_memory.yaml",          None),
    ("AMT-YOLO (no AdapRes)", "configs/amt_yolo_base.yaml",               None),
    ("AMT-YOLO Full",          "configs/amt_yolo_base.yaml",               None),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/amt_yolo_base.yaml")
    parser.add_argument("--dataset", type=str, default="mot17", choices=["coco", "mot17", "visdrone"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--output", type=str, default="experiments/results/ablation.csv")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def evaluate_single(config_path: str, dataset: str, weights_path: str = None) -> dict:
    """Evaluate a single model configuration. Returns metrics dict."""
    # TODO: Full implementation in Phase 5
    # Placeholder returns for structure demonstration
    return {
        "mAP@50": 0.0,
        "mAP@50-95": 0.0,
        "HOTA": 0.0,
        "MOTA": 0.0,
        "IDF1": 0.0,
        "ADE": 0.0,
        "FDE": 0.0,
        "FPS": 0.0,
        "Params_M": 0.0,
        "GFLOPs": 0.0,
    }


def run_ablation(output_path: str):
    """Run all ablation configs and save comparison table."""
    results = []

    for name, config, _ in ABLATION_CONFIGS:
        console.print(f"[cyan]Evaluating:[/cyan] {name}")
        metrics = evaluate_single(config, "mot17")
        metrics["Model"] = name
        results.append(metrics)

    # Print table
    table = Table(title="AMT-YOLO Ablation Study Results")
    cols = ["Model", "mAP@50", "mAP@50-95", "HOTA", "MOTA", "IDF1", "ADE", "FDE", "FPS"]
    for col in cols:
        table.add_column(col, style="cyan" if col == "Model" else "green")

    for r in results:
        table.add_row(*[str(r.get(c, "-")) for c in cols])

    console.print(table)

    # Save CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(results)
    console.print(f"[green]✅ Saved: {output_path}[/green]")


def main():
    args = parse_args()
    if args.ablation:
        run_ablation(args.output)
    else:
        metrics = evaluate_single(args.config, args.dataset, args.weights)
        for k, v in metrics.items():
            console.print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
