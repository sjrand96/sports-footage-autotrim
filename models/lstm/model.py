"""Temporal head for playing / inactive classification on cached features."""

from __future__ import annotations

import torch
import torch.nn as nn

WINDOW_SIZE = 90
WINDOW_RADIUS = WINDOW_SIZE // 2
# Offsets are built as [-WINDOW_RADIUS, ..., -1, 0, 1, ..., WINDOW_RADIUS-1] (len=WINDOW_SIZE),
# so the 0-offset (target frame) sits at index WINDOW_RADIUS.
CENTER_INDEX = WINDOW_RADIUS


class TemporalPlayingClassifier(nn.Module):
    """BiLSTM over a 30-step feature window; predicts label at the center frame."""

    def __init__(
        self,
        feat_dim: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_dropout: float = 0.35,
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
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` (B, 30, D) → logits (B, 1)."""
        if seq.ndim != 3 or seq.shape[1] != WINDOW_SIZE:
            raise ValueError(f"expected (B, {WINDOW_SIZE}, D), got {tuple(seq.shape)}")
        out, _ = self.lstm(seq)
        center = out[:, CENTER_INDEX, :]
        return self.head(self.head_dropout(center))
