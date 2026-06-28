"""
trajectory_labels.py — MOT17 Trajectory Ground Truth Parser
============================================================
Parses MOT17 ground truth files and generates future trajectory
labels for each tracked object in a sequence.

Format of MOT17 gt.txt:
frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, vis
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

class TrajectoryLabelGenerator:
    """
    Parses MOT17 ground truth and generates future trajectory targets.
    """

    def __init__(self, gt_file: str, horizon: int = 5, min_visibility: float = 0.25):
        """
        Args:
            gt_file: Path to MOT17 gt.txt
            horizon: Number of future frames to predict
            min_visibility: Minimum visibility threshold to consider valid
        """
        self.gt_file = Path(gt_file)
        self.horizon = horizon
        self.min_vis = min_visibility
        
        # Internal storage
        # dict mapping frame_idx -> dict mapping obj_id -> (cx, cy, w, h)
        self.frame_to_objects: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        # dict mapping obj_id -> list of (frame_idx, box)
        self.object_tracks: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        
        self.max_frame = 0
        self._parse_gt()

    def _parse_gt(self):
        """Parse the MOT17 gt.txt file."""
        if not self.gt_file.exists():
            # For testing/inference if GT is missing
            return
            
        with open(self.gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                    
                frame_id = int(parts[0])
                obj_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                conf = float(parts[6])
                class_id = int(parts[7])
                vis = float(parts[8])
                
                # In MOT17, class 1 is pedestrian. We only care about pedestrians
                # and objects with sufficient visibility.
                if class_id != 1 or vis < self.min_vis:
                    continue
                    
                # Convert to (cx, cy, w, h)
                cx = bb_left + bb_width / 2.0
                cy = bb_top + bb_height / 2.0
                box = np.array([cx, cy, bb_width, bb_height], dtype=np.float32)
                
                self.frame_to_objects[frame_id][obj_id] = box
                self.object_tracks[obj_id].append((frame_id, box))
                
                if frame_id > self.max_frame:
                    self.max_frame = frame_id
                    
        # Sort object tracks by frame
        for obj_id in self.object_tracks:
            self.object_tracks[obj_id].sort(key=lambda x: x[0])

    def get_detections(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all detection boxes and their IDs for a specific frame.
        
        Returns:
            boxes: [N, 4] array of (cx, cy, w, h)
            ids: [N] array of object IDs
        """
        objects = self.frame_to_objects.get(frame_id, {})
        if not objects:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
            
        ids = list(objects.keys())
        boxes = [objects[i] for i in ids]
        
        return np.stack(boxes), np.array(ids, dtype=np.int32)

    def get_trajectory_targets(self, frame_id: int, obj_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get future trajectory targets for given objects at a specific frame.
        
        Args:
            frame_id: Current frame
            obj_ids: [N] array of object IDs detected in current frame
            
        Returns:
            trajectories: [N, horizon, 4] future boxes (cx, cy, w, h)
            valid_mask: [N, horizon] boolean mask indicating if GT exists for that step
        """
        N = len(obj_ids)
        trajectories = np.zeros((N, self.horizon, 4), dtype=np.float32)
        valid_mask = np.zeros((N, self.horizon), dtype=bool)
        
        for i, obj_id in enumerate(obj_ids):
            # Look ahead for `horizon` frames
            for step in range(1, self.horizon + 1):
                future_frame = frame_id + step
                
                if future_frame > self.max_frame:
                    break
                    
                future_objects = self.frame_to_objects.get(future_frame, {})
                if obj_id in future_objects:
                    trajectories[i, step-1] = future_objects[obj_id]
                    valid_mask[i, step-1] = True
                else:
                    # Object disappeared/occluded.
                    # We pad with the last known position (or current frame's position)
                    # but mark it as invalid in the mask.
                    if step == 1:
                        last_pos = self.frame_to_objects[frame_id][obj_id]
                    else:
                        last_pos = trajectories[i, step-2]
                        
                    trajectories[i, step-1] = last_pos
                    valid_mask[i, step-1] = False
                    
        return trajectories, valid_mask
