"""
train.py — AMT-YOLO Training Script
=====================================
Multi-stage training pipeline:
  Stage 1: Neck Warmup on MOT17 (Backbone & Head frozen)
  Stage 2: Full AMT-YOLO training on MOT17 (all 3 modules)

Optimized for RTX 4050 6GB VRAM:
  - AMP (FP16) enabled by default
  - Gradient accumulation (effective batch = 16)
  - Gradient checkpointing for temporal sequences
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

from amt_yolo.models.amt_yolo import AMTYOLO
from amt_yolo.data.dataset import MOT17SequenceDataset, collate_sequence_batch
from amt_yolo.losses.detection_loss import AMTDetectionLoss
from amt_yolo.losses.trajectory_loss import TrajectoryLoss

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="AMT-YOLO Training Script")
    parser.add_argument(
        "--config", type=str, default="configs/amt_yolo_small.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--stage", type=str, choices=["warmup", "full"], default="full",
        help="Training stage: 'warmup' or 'full'"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify setup without training"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs from config (useful for quick test runs)"
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
            console.print("[yellow]WARNING: < 8GB VRAM detected - using AMP + gradient accumulation[/yellow]")
    return dev


def build_model(cfg, device: torch.device, stage: str):
    model = AMTYOLO(
        backbone=cfg.model.backbone,
        memory_type=cfg.model.memory_type if stage == "full" else "none",
        trajectory_horizon=cfg.model.trajectory_horizon if stage == "full" else 0,
        adaptive_resolution=cfg.model.adaptive_resolution,
        resolutions=list(cfg.model.resolutions),
        pretrained=cfg.model.pretrained,
    ).to(device)
    
    if stage == "warmup":
        # Stage 1: Freeze Backbone AND Detection Head. Only train Neck.
        model.freeze_backbone()
        for param in model.backbone.yolo_model.model[-1].parameters():  # Freeze Detect head
            param.requires_grad = False
        console.print("[yellow]Stage 1 (Warmup): Backbone and Head are FROZEN.[/yellow]")
    else:
        # Stage 2: Unfreeze everything
        model.unfreeze_backbone()
        console.print("[green]Stage 2 (Full): All layers trainable.[/green]")

    console.print(f"[green]Model:[/green] {model.backbone_name} | "
                  f"memory={model.memory_type} | "
                  f"params={model.num_parameters:,}")
    return model


def build_dataloaders(cfg):
    train_dataset = MOT17SequenceDataset(
        root_dir=cfg.data.mot17_path,
        split='train',
        seq_length=cfg.data.sequence_length,
        horizon=cfg.model.trajectory_horizon,
        transform=None  # We'll add augmentation later
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.workers,
        collate_fn=collate_sequence_batch,
        pin_memory=True
    )
    return train_loader


def train_epoch(
    model: AMTYOLO,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    det_loss_fn: AMTDetectionLoss,
    traj_loss_fn: TrajectoryLoss,
    cfg,
    device: torch.device,
    epoch: int,
):
    model.train()
    total_epoch_loss = 0.0
    accumulate_steps = cfg.training.accumulate_grad_batches
    optimizer.zero_grad()
    
    w_det = cfg.loss.detection_weight
    w_traj = cfg.loss.trajectory_weight

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]Epoch {epoch}", total=len(dataloader))

        for step, batch in enumerate(dataloader):
            # batch['frames'] is [B, T, C, H, W]
            frames = batch["frames"].to(device)
            B, T, C, H, W = frames.shape
            
            # Reset memory at start of sequence batch
            model.reset_memory()
            
            batch_loss = torch.tensor(0.0, device=device)
            
            # AMP forward pass sequentially through time
            with torch.amp.autocast('cuda', enabled=cfg.training.amp):
                for t in range(T):
                    frame_t = frames[:, t]
                    
                    # Target parsing for time t
                    # det_labels[batch_idx] is [num_objects, 5] (class, cx, cy, w, h)
                    # We need to convert it to YOLO target format: [N, 6] (batch_idx, class, cx, cy, w, h)
                    yolo_targets = []
                    det_boxes_list = []
                    for b_idx in range(B):
                        if len(batch['det_labels'][b_idx]) > t:
                            labels = batch['det_labels'][b_idx][t].to(device) # [N, 5]
                            if len(labels) > 0:
                                b_col = torch.full((len(labels), 1), b_idx, device=device, dtype=torch.float32)
                                yolo_targets.append(torch.cat([b_col, labels], dim=1))
                                det_boxes_list.append(labels[:, 1:5]) # (cx, cy, w, h)
                                
                    if len(yolo_targets) > 0:
                        yolo_targets = torch.cat(yolo_targets, dim=0)
                        det_boxes = torch.cat(det_boxes_list, dim=0)
                    else:
                        yolo_targets = torch.zeros((0, 6), device=device)
                        det_boxes = torch.zeros((0, 4), device=device)
                    
                    # Forward pass for frame t
                    # Using det_boxes as proxy for current_boxes to TrajectoryHead
                    outputs = model(
                        frame_t, 
                        current_boxes=det_boxes if len(det_boxes) > 0 else None,
                        batch_idx=yolo_targets[:, 0].long() if len(yolo_targets) > 0 else None,
                        return_trajectory=(model.trajectory_head is not None)
                    )
                    
                    # Detection Loss
                    det_loss_dict = det_loss_fn(
                        outputs['detections'], 
                        yolo_targets, 
                        outputs.get('embeddings')
                    )
                    
                    step_loss = w_det * det_loss_dict['total']
                    
                    # Trajectory Loss
                    if outputs.get('trajectory') is not None:
                        # Flatten traj targets across batch
                        traj_targets = []
                        traj_masks = []
                        for b_idx in range(B):
                            if len(batch['traj_labels'][b_idx]) > t:
                                t_labels = batch['traj_labels'][b_idx][t].to(device)
                                t_masks = batch['traj_masks'][b_idx][t].to(device)
                                if len(t_labels) > 0:
                                    traj_targets.append(t_labels)
                                    traj_masks.append(t_masks)
                        
                        if len(traj_targets) > 0:
                            traj_targets = torch.cat(traj_targets, dim=0)
                            traj_masks = torch.cat(traj_masks, dim=0)
                            
                            traj_loss_dict = traj_loss_fn(
                                outputs['trajectory'], 
                                traj_targets, 
                                traj_masks
                            )
                            step_loss += w_traj * traj_loss_dict['total']
                            
                    # Accumulate loss over time
                    batch_loss += step_loss
                
                # Average loss over sequence length
                batch_loss = batch_loss / T

            # Scale and accumulate gradients
            scaler.scale(batch_loss / accumulate_steps).backward()

            # Update weights every accumulate_steps
            if (step + 1) % accumulate_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_epoch_loss += batch_loss.item()
            
            # Progress bar info
            progress.update(task, advance=1, description=f"[cyan]Epoch {epoch} | Loss: {batch_loss.item():.4f}")

    return total_epoch_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    console.print(f"[green]Checkpoint saved:[/green] {path}")


def main():
    args = parse_args()
    console.rule("[bold cyan]AMT-YOLO Training Pipeline")

    # Load config — handle _base_ inheritance manually
    # OmegaConf does not resolve _base_ automatically
    cfg_override = OmegaConf.load(args.config)
    if "_base_" in cfg_override:
        base_path = Path(args.config).parent / cfg_override["_base_"]
        cfg_base = OmegaConf.load(str(base_path))
        # Remove the _base_ key before merging
        cfg_override_clean = OmegaConf.masked_copy(
            cfg_override,
            [k for k in cfg_override.keys() if k != "_base_"]
        )
        cfg = OmegaConf.merge(cfg_base, cfg_override_clean)
    else:
        cfg = cfg_override
    console.print(f"[green]Config:[/green] {args.config}")
    console.print(f"[green]Stage:[/green] {args.stage}")

    # Setup
    device = setup_device(args.device)
    model = build_model(cfg, device, args.stage)

    if args.dry_run:
        console.print("[yellow]Dry run complete — no training started.[/yellow]")
        return

    # Data
    train_loader = build_dataloaders(cfg)

    # Optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.training.amp)
    
    # Losses
    det_loss_fn = AMTDetectionLoss(model.backbone.yolo_model).to(device)
    traj_loss_fn = TrajectoryLoss().to(device)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # strict=False allows loading warmup weights into the full model (which has new modules)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            console.print("[yellow]Optimizer state mismatched (likely due to stage transition). Starting with fresh optimizer.[/yellow]")
            
        start_epoch = ckpt["epoch"] + 1
        console.print(f"[green]Resumed from epoch {start_epoch}[/green]")

    # Training loop
    console.rule("[bold]Starting Training")
    num_epochs = args.epochs if args.epochs is not None else cfg.training.epochs
    train_loss = 0.0
    for epoch in range(start_epoch, num_epochs):
        console.print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, 
            det_loss_fn, traj_loss_fn, cfg, device, epoch
        )
        
        console.print(f"[green]Epoch {epoch+1} Avg Loss:[/green] {train_loss:.4f}")
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % cfg.logging.save_checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, epoch, loss=train_loss,
                path=f"{cfg.paths.checkpoints}/epoch_{epoch+1:03d}.pt",
            )

    # Save final model
    save_checkpoint(
        model, optimizer, num_epochs - 1, loss=train_loss,
        path=f"{cfg.paths.checkpoints}/final_{args.stage}.pt",
    )
    console.rule("[bold green]Training Complete")


if __name__ == "__main__":
    main()
