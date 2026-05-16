"""Pluggable per-frame visual encoders for feature extraction."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DEFAULT_BACKBONE = "efficientnet_v2_m"


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick best available device: cuda → mps → cpu, unless ``requested`` is set."""
    if requested is not None:
        req = requested.strip().lower()
        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if req == "mps":
            if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise RuntimeError("MPS requested but not available (need Apple Silicon + recent PyTorch)")
            return torch.device("mps")
        if req == "cpu":
            return torch.device("cpu")
        if req == "cuda":
            return torch.device("cuda")
        raise ValueError(f"unknown device {requested!r}; use cuda, mps, or cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FrameEncoder(Protocol):
    """Encode video frames into fixed-size feature vectors."""

    name: str
    feat_dim: int
    img_size: int
    device: torch.device

    def preprocess_batch_bgr(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        """BGR frames → normalized batch (B, 3, H, W) on encoder device."""

    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` (B, 3, H, W) on encoder device → (B, feat_dim)."""


class EfficientNetV2Encoder:
    """Torchvision EfficientNetV2 trunk (features + avgpool), no classifier."""

    _VARIANTS: dict[str, tuple[str, object]] = {
        "efficientnet_v2_s": ("efficientnet_v2_s", EfficientNet_V2_S_Weights.IMAGENET1K_V1),
        "efficientnet_v2_m": ("efficientnet_v2_m", EfficientNet_V2_M_Weights.IMAGENET1K_V1),
        "efficientnet_v2_l": ("efficientnet_v2_l", EfficientNet_V2_L_Weights.IMAGENET1K_V1),
    }

    def __init__(self, name: str, *, device: torch.device | None = None) -> None:
        if name not in self._VARIANTS:
            raise ValueError(f"unknown EfficientNetV2 variant: {name!r}")
        ctor_name, weights = self._VARIANTS[name]
        self.name = name
        self.device = device or resolve_device()

        model_fn = getattr(models, ctor_name)
        self._model = model_fn(weights=weights)
        self.feat_dim = int(self._model.classifier[1].in_features)
        self._model.classifier = nn.Identity()
        self._model.eval()
        self._model.to(self.device)

        # Use official preprocessing sizes from weights.transforms(), NOT meta["min_size"]
        # (min_size is the minimum spatial dim the architecture supports, e.g. 33 — not train resolution).
        tfm = weights.transforms()
        crop = tfm.crop_size
        if isinstance(crop, (list, tuple)):
            self.img_size = int(crop[0])
        else:
            self.img_size = int(crop)

        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

    def preprocess_batch_bgr(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        """Resize on CPU, one GPU transfer per batch (faster than per-frame)."""
        import cv2

        if not frames_bgr:
            raise ValueError("empty frame list")

        h = w = self.img_size
        batch = np.empty((len(frames_bgr), h, w, 3), dtype=np.uint8)
        for i, frame in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != h or rgb.shape[1] != w:
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            batch[i] = rgb

        x = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        x = x.to(self.device, non_blocking=self.device.type == "cuda")
        return (x - self._mean) / self._std

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Single-frame helper; prefer ``preprocess_batch_bgr`` for throughput."""
        return self.preprocess_batch_bgr([frame_bgr])[0]

    @torch.inference_mode()
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNetV2 trunk."""
        if images.device != self.device:
            images = images.to(self.device)
        return self._model(images)


_ENCODERS: dict[str, type[EfficientNetV2Encoder]] = {
    "efficientnet_v2_s": EfficientNetV2Encoder,
    "efficientnet_v2_m": EfficientNetV2Encoder,
    "efficientnet_v2_l": EfficientNetV2Encoder,
}


def get_encoder(
    name: str = _DEFAULT_BACKBONE,
    *,
    device: torch.device | str | None = None,
) -> EfficientNetV2Encoder:
    """Construct a registered frame encoder by name."""
    if name not in _ENCODERS:
        known = ", ".join(sorted(_ENCODERS))
        raise ValueError(f"unknown backbone {name!r}; known: {known}")
    dev = resolve_device(device) if isinstance(device, str) else (device or resolve_device())
    return _ENCODERS[name](name, device=dev)


def default_backbone() -> str:
    return _DEFAULT_BACKBONE
