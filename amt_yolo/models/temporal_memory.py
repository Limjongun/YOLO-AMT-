"""
temporal_memory.py — Temporal Memory Module
============================================
Provides inter-frame feature continuity for AMT-YOLO using
recurrent convolutional memory cells.

Two implementations (both benchmarked per research design):
    1. ConvGRU  — faster, less memory, suitable for RTX 4050 6GB
    2. ConvLSTM — more expressive, heavier, better for long sequences

Usage:
    from amt_yolo.models.temporal_memory import TemporalMemoryModule

    memory = TemporalMemoryModule(
        in_channels=256,
        hidden_dim=128,
        mode='convgru',      # or 'convlstm'
    )

    state = None
    for frame_features in sequence:
        enhanced_features, state = memory(frame_features, state)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


# ============================================================
# ConvGRU Cell
# ============================================================

class ConvGRUCell(nn.Module):
    """
    Convolutional Gated Recurrent Unit Cell.
    Processes spatial feature maps recurrently.

    Args:
        in_channels:  Number of input feature channels
        hidden_dim:   Number of hidden state channels
        kernel_size:  Convolution kernel size (default: 3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        # Reset gate
        self.reset_gate = nn.Conv2d(
            in_channels + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=True
        )
        # Update gate
        self.update_gate = nn.Conv2d(
            in_channels + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=True
        )
        # New gate (candidate hidden state)
        self.new_gate = nn.Conv2d(
            in_channels + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C_in, H, W]
            h: Previous hidden state [B, hidden_dim, H, W] or None
        Returns:
            h_new: New hidden state [B, hidden_dim, H, W]
        """
        B, _, H, W = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h], dim=1)  # [B, C_in+hidden, H, W]

        r = torch.sigmoid(self.reset_gate(combined))   # Reset gate
        z = torch.sigmoid(self.update_gate(combined))  # Update gate

        combined_r = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.new_gate(combined_r))       # New gate

        h_new = (1 - z) * h + z * n
        return h_new


# ============================================================
# ConvLSTM Cell
# ============================================================

class ConvLSTMCell(nn.Module):
    """
    Convolutional Long Short-Term Memory Cell.
    More expressive than ConvGRU but uses more memory.
    State: (h, c) — hidden + cell state.

    Args:
        in_channels:  Number of input feature channels
        hidden_dim:   Number of hidden state channels
        kernel_size:  Convolution kernel size (default: 3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        # All 4 gates in one convolution for efficiency
        self.gates = nn.Conv2d(
            in_channels + hidden_dim,
            4 * hidden_dim,  # i, f, g, o gates
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input features [B, C_in, H, W]
            state: (h, c) tuple or None
        Returns:
            h_new: New hidden state [B, hidden_dim, H, W]
            (h_new, c_new): Updated state tuple
        """
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)  # [B, 4*hidden, H, W]

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # Input gate
        f = torch.sigmoid(f)   # Forget gate
        g = torch.tanh(g)      # Cell gate
        o = torch.sigmoid(o)   # Output gate

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


# ============================================================
# Temporal Memory Module (wrapper)
# ============================================================

class TemporalMemoryModule(nn.Module):
    """
    Temporal Memory Module for AMT-YOLO.
    Wraps ConvGRU or ConvLSTM and adds:
        - Input projection (match backbone feature dim)
        - Output projection
        - Feature fusion (residual + gated)
        - Memory reset mechanism

    Args:
        in_channels:    Feature channels from backbone (e.g. 256 for YOLOv8s)
        hidden_dim:     Memory hidden state channels (default: 128, lighter for 6GB)
        mode:           'convgru' | 'convlstm'
        kernel_size:    Memory cell kernel size
        reset_every_n:  Reset memory state every N frames (0 = never reset)
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 128,
        mode: str = "convgru",
        kernel_size: int = 3,
        reset_every_n: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.reset_every_n = reset_every_n
        self._frame_count = 0

        # Input projection: align backbone channels to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Memory cell
        if mode == "convgru":
            self.cell = ConvGRUCell(hidden_dim, hidden_dim, kernel_size)
        elif mode == "convlstm":
            self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'convgru' or 'convlstm'.")

        # Output projection: back to in_channels
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # Gated fusion: balance between current frame and memory
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        self._state = None  # Internal memory state

    def reset_memory(self):
        """Manually reset memory state (call between sequences)."""
        self._state = None
        self._frame_count = 0

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Union[torch.Tensor, Tuple]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple]]:
        """
        Args:
            x:     Current feature map from backbone [B, C, H, W]
            state: Previous memory state (or None for first frame)
                   - ConvGRU:  Tensor [B, hidden_dim, H, W]
                   - ConvLSTM: Tuple (h, c) each [B, hidden_dim, H, W]
        Returns:
            enhanced: Memory-enhanced feature [B, C, H, W]
            new_state: Updated memory state
        """
        # Auto-reset based on frame counter
        self._frame_count += 1
        if self.reset_every_n > 0 and self._frame_count % self.reset_every_n == 0:
            state = None

        # Project input
        projected = self.input_proj(x)  # [B, hidden_dim, H, W]

        # Update memory
        if self.mode == "convgru":
            h = self.cell(projected, state)
            new_state = h
            memory_out = h
        else:  # convlstm
            h, new_state = self.cell(projected, state)
            memory_out = h

        # Project back to input space
        memory_feat = self.output_proj(memory_out)  # [B, C, H, W]

        # Gated fusion: blend current frame with memory
        gate_input = torch.cat([x, memory_feat], dim=1)
        gate = self.fusion_gate(gate_input)
        enhanced = gate * x + (1 - gate) * memory_feat  # [B, C, H, W]

        # Residual connection
        enhanced = enhanced + x

        return enhanced, new_state

    def get_memory_stats(self) -> dict:
        """Returns memory usage statistics for profiling."""
        return {
            "mode": self.mode,
            "hidden_dim": self.hidden_dim,
            "frame_count": self._frame_count,
            "has_state": self._state is not None,
        }
