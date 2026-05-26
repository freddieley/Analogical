"""Abstract base classes for the simulation layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SensoryBundle:
    """Multimodal sensory observation produced by the environment at each step.

    All modalities are optional; missing ones are represented as ``None``.
    The agent must handle any combination gracefully (perception-first design).
    """

    proprioception: Optional[np.ndarray] = None  # (proprioception_dim,) float32
    vision: Optional[np.ndarray] = None          # (1, H, W) float32 in [0, 1]
    audio: Optional[np.ndarray] = None           # (samples,) float32
    tactile: Optional[np.ndarray] = None         # (n_contacts,) float32
    timestamp_ms: float = 0.0                    # wall-clock ms when bundle was collected
    metadata: Dict[str, Any] = field(default_factory=dict)

    def available_modalities(self) -> Dict[str, np.ndarray]:
        """Return dict of non-None modalities."""
        result: Dict[str, np.ndarray] = {}
        if self.proprioception is not None:
            result["proprioception"] = self.proprioception
        if self.vision is not None:
            result["vision"] = self.vision
        if self.audio is not None:
            result["audio"] = self.audio
        if self.tactile is not None:
            result["tactile"] = self.tactile
        return result

    def clone(self) -> "SensoryBundle":
        return SensoryBundle(
            proprioception=None if self.proprioception is None else self.proprioception.copy(),
            vision=None if self.vision is None else self.vision.copy(),
            audio=None if self.audio is None else self.audio.copy(),
            tactile=None if self.tactile is None else self.tactile.copy(),
            timestamp_ms=self.timestamp_ms,
            metadata=dict(self.metadata),
        )


class BaseEnv(ABC):
    """Minimal interface every simulator must implement."""

    @abstractmethod
    def reset(self) -> SensoryBundle:
        """Reset environment to initial conditions; return first observation."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[SensoryBundle, float, bool, dict]:
        """Apply *action* and return (bundle, reward, done, info)."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the continuous action vector."""
        ...

    @property
    @abstractmethod
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) arrays of shape (action_dim,)."""
        ...

    @property
    @abstractmethod
    def proprioception_dim(self) -> int:
        """Dimensionality of the proprioceptive state vector."""
        ...

    @property
    def task_name(self) -> str:
        return self.__class__.__name__
