"""Temporal head for playing / inactive classification on cached features."""

from __future__ import annotations

import torch
import torch.nn as nn

WINDOW_SIZE = 30
CENTER_INDEX = 15  # T-15 .. T+14 → center at 15


class TemporalPlayingClassifier(nn.Module):
    """BiLSTM over a 30-step feature window; predicts label at the center frame."""

    def __init__(
        self,
        feat_dim: int,
        *,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` (B, 30, D) → logits (B, 1)."""
        if seq.ndim != 3 or seq.shape[1] != WINDOW_SIZE:
            raise ValueError(f"expected (B, {WINDOW_SIZE}, D), got {tuple(seq.shape)}")
        out, _ = self.lstm(seq)
        center = out[:, CENTER_INDEX, :]
        return self.head(center)
