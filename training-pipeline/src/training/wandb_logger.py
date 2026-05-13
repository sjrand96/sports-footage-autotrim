"""Thin wrapper around Weights & Biases logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class WandbConfig:
    project: str
    run_name: Optional[str] = None
    enabled: bool = True


class WandbLogger:
    def __init__(self, cfg: WandbConfig, config: Dict[str, Any]) -> None:
        self.enabled = cfg.enabled
        self._wandb = None
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            self.enabled = False
            return
        self._wandb = wandb
        self._wandb.init(project=cfg.project, name=cfg.run_name, config=config)

    def get(self) -> Any:
        return self._wandb

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._wandb:
            self._wandb.log(metrics, step=step)

    def log_confusion_matrix(self, y_true: List[int], y_pred: List[int], labels: List[str]) -> None:
        if not (self.enabled and self._wandb):
            return
        try:
            plot = self._wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=labels,
            )
            self._wandb.log({"confusion_matrix": plot})
        except Exception:
            counts = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(y_true, y_pred):
                counts[t][p] += 1
            self._wandb.log({"confusion_matrix_counts": self._wandb.Table(data=counts, columns=labels)})

    def log_table(self, name: str, columns: List[str], data: Iterable[List[Any]]) -> None:
        if not (self.enabled and self._wandb):
            return
        table = self._wandb.Table(columns=columns, data=list(data))
        self._wandb.log({name: table})

    def log_media(self, name: str, media: Any) -> None:
        if not (self.enabled and self._wandb):
            return
        self._wandb.log({name: media})

    def finish(self) -> None:
        if self.enabled and self._wandb:
            self._wandb.finish()
