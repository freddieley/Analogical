"""Per-modality sensory encoders.

Each encoder maps a raw sensory array to a shared d_model-dimensional embedding.
All weights are randomly initialised (zero pretraining) and may be updated
online by the PlasticUpdater.

Encoders handle missing inputs gracefully: a missing modality returns a zero
embedding and signals reliability = 0.0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseEncoder(nn.Module):
    """Shared contract for all modality encoders."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self, x: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding [d_model], reliability [scalar]) — reliability 0 if x is None."""
        raise NotImplementedError


class ProprioceptionEncoder(_BaseEncoder):
    """Linear projection with layer-norm. Handles variable-dim input."""

    def __init__(self, input_dim: int, d_model: int = 256) -> None:
        super().__init__(d_model)
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x is None:
            zeros = torch.zeros(self.d_model)
            return zeros, torch.tensor(0.0)
        emb = self.net(x)
        return emb, torch.tensor(1.0)


class VisionEncoder(_BaseEncoder):
    """Small conv stack for (1, 64, 64) greyscale frames.

    Architecture: 3-layer conv → flatten → linear.
    This deliberately avoids any pretrained backbone.
    """

    def __init__(self, d_model: int = 256, img_channels: int = 1) -> None:
        super().__init__(d_model)
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 16, kernel_size=8, stride=4),  # → 16×15×15
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),             # → 32×6×6
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),             # → 64×4×4
            nn.GELU(),
        )
        conv_out = 64 * 4 * 4  # 1024
        self.proj = nn.Sequential(
            nn.Linear(conv_out, d_model),
            nn.LayerNorm(d_model),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x is None:
            zeros = torch.zeros(self.d_model)
            return zeros, torch.tensor(0.0)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # add batch dim
        feats = self.conv(x)
        flat = feats.flatten(start_dim=1)
        emb = self.proj(flat).squeeze(0)
        return emb, torch.tensor(1.0)


class TactileEncoder(_BaseEncoder):
    """Small MLP for contact/force signals."""

    def __init__(self, input_dim: int = 1, d_model: int = 256) -> None:
        super().__init__(d_model)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x is None:
            return torch.zeros(self.d_model), torch.tensor(0.0)
        return self.net(x), torch.tensor(1.0)


class AudioEncoder(_BaseEncoder):
    """1D conv encoder for raw audio signals (optional modality)."""

    def __init__(
        self, window_size: int = 512, d_model: int = 256
    ) -> None:
        super().__init__(d_model)
        self.window_size = window_size
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = nn.Linear(32 * 8, d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x is None:
            return torch.zeros(self.d_model), torch.tensor(0.0)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        feats = self.conv(x)
        flat = feats.flatten(start_dim=1)
        emb = self.proj(flat).squeeze(0)
        return emb, torch.tensor(1.0)
