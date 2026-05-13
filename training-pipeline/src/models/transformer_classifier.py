"""Transformer encoder classifier for window classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TransformerConfig:
    input_dim: int
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    max_len: int = 64


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:length]


class TransformerClassifier(nn.Module):
    def __init__(self, cfg: TransformerConfig, num_classes: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(cfg.input_dim, cfg.model_dim)
        self.pos_enc = PositionalEncoding(cfg.model_dim, cfg.max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.model_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.model_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
