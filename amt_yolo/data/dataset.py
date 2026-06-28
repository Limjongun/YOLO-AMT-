"""
dataset.py — MOT17 Temporal Sequence Dataset
=============================================
Loads consecutive frames from MOT17 sequences for temporal training.
Provides image tensors, bounding boxes, and future trajectory labels.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from amt_yolo.data.trajectory_labels import TrajectoryLabelGenerator

# Target resolution — keeps VRAM manageable on RTX 4050 6GB
# Original MOT17 frames are 1080x1920 which would OOM at batch_size=2
TARGET_SIZE = (640, 640)  # (height, width)

class MOT17SequenceDataset(Dataset):
    """
    Dataset for MOT17 that yields sequences of frames instead of independent images.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        seq_length: int = 5,
        horizon: int = 5,
        transform=None,
    ):
        """
        Args:
            root_dir: Path to MOT17 directory (e.g. D:/YOLO next/datasets/MOT17)
            split: 'train' or 'test'
            seq_length: Number of consecutive frames per sequence batch
            horizon: Number of future frames to predict for trajectory
            transform: Temporal augmentation pipeline
        """
        self.root_dir = Path(root_dir) / split
        self.seq_length = seq_length
        self.horizon = horizon
        self.transform = transform
        
        self.sequences = []  # List of dicts representing valid sequences
        
        self._load_dataset()

    def _load_dataset(self):
        """Scan MOT17 directory and build sequence indices."""
        if not self.root_dir.exists():
            print(f"Warning: {self.root_dir} not found. Returning empty dataset.")
            return

        # MOT17 subdirectories: MOT17-02-FRCNN, MOT17-02-SDP, etc.
        seq_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        for seq_dir in seq_dirs:
            img_dir = seq_dir / 'img1'
            gt_file = seq_dir / 'gt' / 'gt.txt'
            
            if not img_dir.exists():
                continue
                
            # Initialize Trajectory generator for this sequence
            traj_gen = None
            if gt_file.exists():
                traj_gen = TrajectoryLabelGenerator(str(gt_file), horizon=self.horizon)
                
            # Get sorted list of images
            img_files = sorted([f for f in img_dir.glob('*.jpg')])
            num_frames = len(img_files)
            
            if num_frames < self.seq_length:
                continue
                
            # Create overlapping sequences
            # e.g., if seq_length=5, frames [1,2,3,4,5], [2,3,4,5,6], etc.
            stride = 1 # or larger stride to reduce correlation
            for start_idx in range(0, num_frames - self.seq_length + 1, stride):
                seq_frames = img_files[start_idx : start_idx + self.seq_length]
                frame_ids = [int(f.stem) for f in seq_frames]
                
                self.sequences.append({
                    'seq_name': seq_dir.name,
                    'frame_paths': seq_frames,
                    'frame_ids': frame_ids,
                    'traj_gen': traj_gen
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict containing:
                'frames': Tensor [T, 3, H, W]
                'det_labels': List[Tensor] (one per frame)
                'traj_labels': Tensor [T, N, horizon, 4] (trajectory GT)
                'traj_masks': Tensor [T, N, horizon] (validity mask)
                'reset_mem': True (signals memory to reset for new sequence)
        """
        seq_info = self.sequences[idx]
        frame_paths = seq_info['frame_paths']
        frame_ids = seq_info['frame_ids']
        traj_gen = seq_info['traj_gen']
        
        frames = []
        det_labels = []
        traj_labels = []
        traj_masks = []
        
        # Load images
        for i, (path, f_id) in enumerate(zip(frame_paths, frame_ids)):
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to target size to prevent VRAM OOM (original is 1080x1920)
            img = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(img)
            
            # Load GT if available
            if traj_gen is not None:
                boxes, obj_ids = traj_gen.get_detections(f_id)
                trajectories, valid_mask = traj_gen.get_trajectory_targets(f_id, obj_ids)
                
                # Convert boxes to YOLO format (class, cx, cy, w, h)
                # Note: MOT17 bounding boxes are raw pixels. YOLO expects normalized [0,1]
                h, w, _ = img.shape
                if len(boxes) > 0:
                    norm_boxes = boxes.copy()
                    norm_boxes[:, [0, 2]] /= w
                    norm_boxes[:, [1, 3]] /= h
                    
                    norm_traj = trajectories.copy()
                    norm_traj[:, :, [0, 2]] /= w
                    norm_traj[:, :, [1, 3]] /= h
                    
                    # Add class ID column (Class 0 for pedestrian in YOLO)
                    cls_col = np.zeros((len(boxes), 1), dtype=np.float32)
                    labels = np.hstack([cls_col, norm_boxes])
                else:
                    labels = np.zeros((0, 5), dtype=np.float32)
                    norm_traj = np.zeros((0, self.horizon, 4), dtype=np.float32)
                    valid_mask = np.zeros((0, self.horizon), dtype=bool)
                    
                det_labels.append(labels)
                traj_labels.append(norm_traj)
                traj_masks.append(valid_mask)
            else:
                det_labels.append(np.zeros((0, 5)))
                traj_labels.append(np.zeros((0, self.horizon, 4)))
                traj_masks.append(np.zeros((0, self.horizon), dtype=bool))
                
        # Apply Temporal Augmentations
        if self.transform is not None:
            frames, det_labels, traj_labels = self.transform(frames, det_labels, traj_labels)
            
        # Convert to tensors
        # Frames: [T, H, W, C] -> [T, C, H, W]
        frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        
        return {
            'frames': frames_tensor,
            'det_labels': [torch.from_numpy(l).float() for l in det_labels],
            'traj_labels': [torch.from_numpy(t).float() for t in traj_labels],
            'traj_masks': [torch.from_numpy(m).bool() for m in traj_masks],
            'seq_name': seq_info['seq_name'],
            'reset_mem': True
        }

def collate_sequence_batch(batch):
    """
    Custom collate_fn for sequence batches since det_labels have variable lengths.
    """
    frames = torch.stack([b['frames'] for b in batch]) # [B, T, C, H, W]
    
    # We keep labels as lists of lists: batch -> time -> N objects
    det_labels = [b['det_labels'] for b in batch]
    traj_labels = [b['traj_labels'] for b in batch]
    traj_masks = [b['traj_masks'] for b in batch]
    
    seq_names = [b['seq_name'] for b in batch]
    reset_mems = [b['reset_mem'] for b in batch]
    
    return {
        'frames': frames,
        'det_labels': det_labels,
        'traj_labels': traj_labels,
        'traj_masks': traj_masks,
        'seq_name': seq_names,
        'reset_mem': reset_mems
    }
