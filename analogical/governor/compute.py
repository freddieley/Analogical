"""Compute governor — adaptive budget control for real-time inference.

Monitors per-step wall-clock latency and automatically adjusts MPPI N/H to
stay within the 200 ms hard cap.  Implements graceful CPU-only degradation:
if a GPU OOM or CUDA error occurs mid-episode, the governor transparently
moves computation to CPU without disrupting the control loop.

Budget levels
-------------
FULL    : N=512, H=10  (GPU only — >4 GB VRAM)
REDUCED : N=256, H=8   (GPU or fast CPU)
MINIMAL : N=64,  H=5   (any CPU)
CRISIS  : N=32,  H=3   (emergency: last resort to meet 200 ms cap)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BudgetLevel:
    name: str
    n_samples: int
    horizon: int
    target_ms: float   # target step latency to maintain this level


_LEVELS: list[BudgetLevel] = [
    BudgetLevel("FULL",    512, 10, 120.0),
    BudgetLevel("REDUCED", 256,  8, 150.0),
    BudgetLevel("MINIMAL",  64,  5, 180.0),
    BudgetLevel("CRISIS",   32,  3, 195.0),
]

_SCALE_UP_AFTER = 10    # steps of good latency before upgrading budget
_SCALE_DOWN_AFTER = 3   # consecutive violations before downgrading


class ComputeGovernor:
    """Real-time compute budget controller.

    Parameters
    ----------
    hard_cap_ms : float
        Absolute latency ceiling (200 ms by default).
    history_len : int
        Window of recent step times used for adaptation decisions.
    prefer_gpu  : bool
        Try GPU first; fall back to CPU on failure or OOM.
    """

    def __init__(
        self,
        hard_cap_ms: float = 200.0,
        history_len: int = 20,
        prefer_gpu: bool = True,
    ) -> None:
        self.hard_cap_ms = hard_cap_ms
        self._history: deque[float] = deque(maxlen=history_len)
        self._level_idx = 0  # start at FULL
        self._good_streak = 0
        self._bad_streak = 0

        # Device selection
        self._device = self._select_device(prefer_gpu)

    # ── Device management ─────────────────────────────────────────────────

    @staticmethod
    def _select_device(prefer_gpu: bool) -> torch.device:
        if prefer_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        elif prefer_gpu and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def fallback_to_cpu(self) -> None:
        """Explicitly move to CPU (e.g. after CUDA OOM)."""
        self._device = torch.device("cpu")
        # Also step down to at least MINIMAL
        self._level_idx = max(self._level_idx, 2)

    # ── Budget reporting ──────────────────────────────────────────────────

    def record_step_time(self, elapsed_ms: float) -> None:
        """Record the latency of the last inference+control step."""
        self._history.append(elapsed_ms)
        self._adapt()

    def _adapt(self) -> None:
        if not self._history:
            return
        current_level = _LEVELS[self._level_idx]
        recent_avg = sum(self._history) / len(self._history)

        # Scale down if consistently over target
        if recent_avg > current_level.target_ms:
            self._bad_streak += 1
            self._good_streak = 0
            if self._bad_streak >= _SCALE_DOWN_AFTER:
                self._level_idx = min(self._level_idx + 1, len(_LEVELS) - 1)
                self._bad_streak = 0
        else:
            self._good_streak += 1
            self._bad_streak = 0
            # Scale up if consistently comfortable
            if self._good_streak >= _SCALE_UP_AFTER and self._level_idx > 0:
                self._level_idx -= 1
                self._good_streak = 0

    # ── Query interface ───────────────────────────────────────────────────

    @property
    def current_level(self) -> BudgetLevel:
        return _LEVELS[self._level_idx]

    def get_n_samples(self) -> int:
        return _LEVELS[self._level_idx].n_samples

    def get_horizon(self) -> int:
        return _LEVELS[self._level_idx].horizon

    def remaining_budget_ms(self, elapsed_so_far_ms: float) -> float:
        """Return how many ms remain before the hard cap."""
        return max(0.0, self.hard_cap_ms - elapsed_so_far_ms)

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        lvl = self.current_level
        recent = list(self._history)
        return {
            "device": str(self._device),
            "budget_level": lvl.name,
            "n_samples": lvl.n_samples,
            "horizon": lvl.horizon,
            "recent_avg_ms": round(sum(recent) / len(recent), 2) if recent else None,
            "recent_max_ms": round(max(recent), 2) if recent else None,
            "hard_cap_ms": self.hard_cap_ms,
        }
