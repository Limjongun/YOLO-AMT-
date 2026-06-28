"""
augmentation.py — Temporal-Consistent Augmentations
===================================================
Applies identical augmentations across all frames in a temporal sequence
to ensure spatial consistency (objects don't teleport due to random crop/flip).
"""

import cv2
import numpy as np
import random
from typing import List, Tuple

class TemporalAugmentation:
    """
    Applies consistent transformations across a sequence of frames.
    """
    def __init__(self, flip_prob=0.5, color_jitter_prob=0.5):
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob

    def __call__(
        self, 
        frames: List[np.ndarray], 
        det_labels: List[np.ndarray], 
        traj_labels: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        
        # Determine operations for this specific sequence ONCE
        do_flip = random.random() < self.flip_prob
        do_color = random.random() < self.color_jitter_prob
        
        # Color jitter params
        if do_color:
            alpha = 1.0 + random.uniform(-0.3, 0.3) # Contrast
            beta = random.uniform(-30, 30)          # Brightness
            
        aug_frames = []
        aug_det = []
        aug_traj = []
        
        for i in range(len(frames)):
            img = frames[i]
            det = det_labels[i].copy() if len(det_labels[i]) > 0 else det_labels[i]
            traj = traj_labels[i].copy() if len(traj_labels[i]) > 0 else traj_labels[i]
            
            # 1. Color Jitter (Doesn't affect boxes)
            if do_color:
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                
            # 2. Horizontal Flip
            if do_flip:
                img = cv2.flip(img, 1)
                
                # Flip bounding boxes: x_center = 1.0 - x_center
                if len(det) > 0:
                    det[:, 1] = 1.0 - det[:, 1]
                    
                # Flip trajectory targets: x_center = 1.0 - x_center
                if len(traj) > 0:
                    traj[:, :, 0] = 1.0 - traj[:, :, 0]
            
            aug_frames.append(img)
            aug_det.append(det)
            aug_traj.append(traj)
            
        return aug_frames, aug_det, aug_traj
