"""
train.py — AMT-YOLO Training Script
=====================================
Multi-stage training pipeline:
  Stage 1: Detection pre-training on COCO (YOLOv8 backbone)
  Stage 2: Full AMT-YOLO training on MOT17 (all 3 modules)

Optimized for RTX 4050 6GB VRAM:
  - AMP (FP16) enabled by default
  - Gradient accumulation (effective batch = 16)
  - Gradient checkpointing for temporal sequences
  - Max training resolution: 768x768

Usage:
    # Stage 1 - detection pre-training
    python scripts/train.py --config configs/amt_yolo_small.yaml --stage detection

    # Stage 2 - full training
    python scripts/train.py --config configs/amt_yolo_base.yaml --stage full

    # Resume from checkpoint
    python scripts/train.py --config configs/amt_yolo_base.yaml --resume experiments/checkpoints/last.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="AMT-YOLO Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/amt_yolo_small.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["detection", "full"],
        default="full",
        help="Training stage: 'detection' (COCO pre-train) or 'full' (all modules)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify setup without training",
    )
    return parser.parse_args()


def setup_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"[green]GPU:[/green] {gpu_name} ({vram:.1f} GB VRAM)")
        if vram < 8.0:
            console.print(
                "[yellow]⚠ < 8GB VRAM detected — using AMP + gradient accumulation[/yellow]"
            )
    return dev


def build_model(cfg, device: torch.device):
    from amt_yolo.models.amt_yolo import AMTYOLO

    model = AMTYOLO(
        backbone=cfg.model.backbone,
        memory_type=cfg.model.memory_type,
        trajectory_horizon=cfg.model.trajectory_horizon,
        adaptive_resolution=cfg.model.adaptive_resolution,
        resolutions=list(cfg.model.resolutions),
        pretrained=cfg.model.pretrained,
    ).to(device)

    console.print(f"[green]Model:[/green] {model.backbone_name} | "
                  f"memory={model.memory_type} | "
                  f"horizon={model.trajectory_horizon} | "
                  f"params={model.num_parameters:,}")
    return model


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


def build_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.training.learning_rate * 0.01,
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    scaler: GradScaler,
    cfg,
    device: torch.device,
    epoch: int,
    accumulate_steps: int = 4,
):
    """Single training epoch with AMP + gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]Epoch {epoch}", total=len(dataloader))

        for step, batch in enumerate(dataloader):
            images = batch["images"].to(device)
            targets = batch.get("targets")

            # AMP forward pass
            with autocast(enabled=cfg.training.amp):
                outputs = model(images)
                # TODO: Compute combined loss (detection + trajectory)
                loss = torch.tensor(0.0, requires_grad=True, device=device)

            # Scale and accumulate gradients
            scaler.scale(loss / accumulate_steps).backward()

            # Update weights every accumulate_steps
            if (step + 1) % accumulate_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress.advance(task)

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    console.print(f"[green]✅ Checkpoint saved:[/green] {path}")


def main():
    args = parse_args()

    console.rule("[bold cyan]AMT-YOLO Training")

    # Load config
    cfg = OmegaConf.load(args.config)
    console.print(f"[green]Config:[/green] {args.config}")
    console.print(f"[green]Stage:[/green] {args.stage}")

    # Setup
    device = setup_device(args.device)
    model = build_model(cfg, device)

    if args.dry_run:
        console.print("[yellow]Dry run complete — no training started.[/yellow]")
        return

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg.training.amp)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        console.print(f"[green]Resumed from epoch {start_epoch}[/green]")

    # Training loop
    console.rule("[bold]Starting Training")
    for epoch in range(start_epoch, cfg.training.epochs):
        console.print(f"\nEpoch [{epoch+1}/{cfg.training.epochs}]")

        # TODO: Initialize dataloader (Phase 3)
        # train_loss = train_epoch(model, train_loader, optimizer, scaler, cfg, device, epoch)

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % cfg.logging.save_checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                loss=0.0,  # placeholder
                path=f"{cfg.paths.checkpoints}/epoch_{epoch+1:03d}.pt",
            )

    # Save final model
    save_checkpoint(
        model, optimizer, cfg.training.epochs - 1,
        loss=0.0,
        path=f"{cfg.paths.checkpoints}/final.pt",
    )
    console.rule("[bold green]Training Complete")


if __name__ == "__main__":
    main()
