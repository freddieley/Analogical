"""Benchmark metrics: episode and suite level.

Metrics beyond binary pass/fail
--------------------------------
- task_success           : did the agent complete the task?
- timing_compliance_rate : fraction of steps within 200 ms hard cap
- recovery_latency_ms    : ms from perturbation onset to re-stabilisation
- degradation_auc        : AUC of success rate vs perturbation intensity
                           (higher = more graceful degradation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepRecord:
    step: int
    elapsed_ms: float
    reward: float
    done: bool
    info: dict
    perturbation_label: str = ""


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode run."""

    task_name: str
    task_id: str                            # "A" or "C"
    perturbation_label: str

    # Core metrics
    task_success: bool = False
    total_steps: int = 0
    total_reward: float = 0.0

    # Timing
    step_times_ms: List[float] = field(default_factory=list)
    timing_violations: List[float] = field(default_factory=list)  # steps > 200ms

    # Perturbation recovery
    perturbation_onset_step: Optional[int] = None
    recovery_step: Optional[int] = None  # first stable step after perturbation

    # World model stats
    wm_losses: List[float] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def timing_compliance_rate(self) -> float:
        if not self.step_times_ms:
            return 1.0
        ok = sum(1 for t in self.step_times_ms if t <= 200.0)
        return ok / len(self.step_times_ms)

    @property
    def mean_step_ms(self) -> float:
        if not self.step_times_ms:
            return 0.0
        return sum(self.step_times_ms) / len(self.step_times_ms)

    @property
    def max_step_ms(self) -> float:
        return max(self.step_times_ms, default=0.0)

    @property
    def recovery_latency_ms(self) -> Optional[float]:
        """ms from perturbation onset to first re-stabilised step.
        None if no perturbation occurred or agent never recovered.
        """
        if self.perturbation_onset_step is None or self.recovery_step is None:
            return None
        steps_to_recover = self.recovery_step - self.perturbation_onset_step
        # Assume ~20ms physics step
        return max(0.0, steps_to_recover * 20.0)

    @property
    def mean_wm_loss(self) -> float:
        return sum(self.wm_losses) / len(self.wm_losses) if self.wm_losses else 0.0

    def as_dict(self) -> dict:
        return {
            "task": self.task_name,
            "task_id": self.task_id,
            "perturbation": self.perturbation_label,
            "success": self.task_success,
            "total_steps": self.total_steps,
            "total_reward": round(self.total_reward, 3),
            "timing_compliance": round(self.timing_compliance_rate, 4),
            "mean_step_ms": round(self.mean_step_ms, 2),
            "max_step_ms": round(self.max_step_ms, 2),
            "recovery_latency_ms": (
                round(self.recovery_latency_ms, 1)
                if self.recovery_latency_ms is not None
                else None
            ),
            "mean_wm_loss": round(self.mean_wm_loss, 5),
        }


@dataclass
class SuiteMetrics:
    """Aggregated metrics across all episodes in a benchmark suite."""

    episodes: List[EpisodeMetrics] = field(default_factory=list)

    # ── Aggregate properties ──────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.task_success for e in self.episodes) / len(self.episodes)

    @property
    def timing_compliance_rate(self) -> float:
        if not self.episodes:
            return 1.0
        return sum(e.timing_compliance_rate for e in self.episodes) / len(self.episodes)

    @property
    def mean_recovery_latency_ms(self) -> Optional[float]:
        valid = [
            e.recovery_latency_ms
            for e in self.episodes
            if e.recovery_latency_ms is not None
        ]
        if not valid:
            return None
        return sum(valid) / len(valid)

    @property
    def degradation_auc(self) -> float:
        """AUC of success rate as perturbation intensity increases.

        Episodes are ordered by perturbation severity (approximated by order
        in the suite).  AUC ∈ [0, 1]; higher = more graceful degradation.
        """
        if len(self.episodes) < 2:
            return float(self.episodes[0].task_success) if self.episodes else 0.0

        successes = [float(e.task_success) for e in self.episodes]
        n = len(successes)
        # Normalised trapezoid AUC over [0, 1] x-axis
        auc = 0.0
        dx = 1.0 / (n - 1)
        for i in range(n - 1):
            auc += 0.5 * (successes[i] + successes[i + 1]) * dx
        return round(auc, 4)

    @property
    def per_task_success_rate(self) -> Dict[str, float]:
        """Success rate broken down by task ID (A/C)."""
        by_task: Dict[str, List[bool]] = {}
        for ep in self.episodes:
            by_task.setdefault(ep.task_id, []).append(ep.task_success)
        return {
            tid: sum(v) / len(v) for tid, v in by_task.items()
        }

    def summary(self) -> dict:
        return {
            "n_episodes": len(self.episodes),
            "success_rate": round(self.success_rate, 4),
            "timing_compliance_rate": round(self.timing_compliance_rate, 4),
            "mean_recovery_latency_ms": (
                round(self.mean_recovery_latency_ms, 1)
                if self.mean_recovery_latency_ms is not None
                else None
            ),
            "degradation_auc": self.degradation_auc,
            "per_task_success_rate": {
                k: round(v, 4) for k, v in self.per_task_success_rate.items()
            },
        }

    def per_episode_table(self) -> List[dict]:
        return [e.as_dict() for e in self.episodes]
