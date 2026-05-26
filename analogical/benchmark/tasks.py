"""Benchmark task definitions and perturbation suite builder.

Tasks
-----
BalanceTask (Task C) — inverted pendulum balance under actuator perturbation.
ReachTask   (Task A) — 2-joint arm reaching under sensor dropout/noise.

Both tasks are instantiated with their full perturbation schedule and run
through the BenchmarkRunner to produce SuiteMetrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from analogical.sim.base import BaseEnv
from analogical.sim.mock_env import MockCartpoleEnv, MockReachEnv
from analogical.sim.perturbation import (
    PerturbationConfig,
    PerturbationHarness,
    _ScheduledEvent,
)


@dataclass
class TaskSpec:
    """Lightweight descriptor for one benchmark scenario."""

    name: str            # human-readable
    env: BaseEnv         # (possibly wrapped) environment
    max_steps: int
    success_fn: object   # callable(info) -> bool
    task_id: str         # "A" or "C"
    perturbation_label: str = "none"


def _balance_success(info: dict) -> bool:
    """Episode succeeds if pole never fell (done via max_steps, not failed)."""
    return not info.get("failed", False)


def _reach_success(info: dict) -> bool:
    return info.get("reached", False)


class BalanceTask:
    """Task C — balance under actuator perturbation scenario suite."""

    MAX_STEPS = 500

    @classmethod
    def scenarios(
        cls, seed: Optional[int] = None
    ) -> list[TaskSpec]:
        """Return all perturbation scenarios for Task C."""
        specs = []
        base_env = MockCartpoleEnv(max_steps=cls.MAX_STEPS, seed=seed)

        # Scenario definitions: (label, schedule)
        scenarios = [
            ("baseline", []),
            ("actuator_noise_mild", [
                _ScheduledEvent(50, PerturbationConfig(
                    actuator_noise_sigma=0.3, label="mild_noise", onset_step=50
                ))
            ]),
            ("actuator_noise_heavy", [
                _ScheduledEvent(50, PerturbationConfig(
                    actuator_noise_sigma=0.8, label="heavy_noise", onset_step=50
                ))
            ]),
            ("dropout_30pct", [
                _ScheduledEvent(50, PerturbationConfig(
                    actuator_dropout_prob=0.3, label="dropout_30pct", onset_step=50
                ))
            ]),
            ("dropout_60pct", [
                _ScheduledEvent(50, PerturbationConfig(
                    actuator_dropout_prob=0.6, label="dropout_60pct", onset_step=50
                ))
            ]),
            ("force_kick", [
                _ScheduledEvent(80, PerturbationConfig(
                    force_kick_magnitude=2.5, label="force_kick", onset_step=80
                ))
            ]),
            ("escalating_dropout", [
                _ScheduledEvent(0,   PerturbationConfig(label="clean")),
                _ScheduledEvent(100, PerturbationConfig(actuator_dropout_prob=0.2, label="ramp_20")),
                _ScheduledEvent(200, PerturbationConfig(actuator_dropout_prob=0.4, label="ramp_40")),
                _ScheduledEvent(300, PerturbationConfig(actuator_dropout_prob=0.6, label="ramp_60")),
            ]),
            ("noise_plus_dropout", [
                _ScheduledEvent(60, PerturbationConfig(
                    actuator_dropout_prob=0.3,
                    actuator_noise_sigma=0.4,
                    label="combined",
                    onset_step=60,
                ))
            ]),
        ]

        for label, schedule in scenarios:
            env = PerturbationHarness(
                MockCartpoleEnv(max_steps=cls.MAX_STEPS, seed=seed),
                schedule=schedule,
                seed=seed,
            )
            specs.append(TaskSpec(
                name=f"balance/{label}",
                env=env,
                max_steps=cls.MAX_STEPS,
                success_fn=_balance_success,
                task_id="C",
                perturbation_label=label,
            ))
        return specs


class ReachTask:
    """Task A — reaching under sensor dropout/noise scenario suite."""

    MAX_STEPS = 400

    @classmethod
    def scenarios(
        cls, seed: Optional[int] = None
    ) -> list[TaskSpec]:
        """Return all perturbation scenarios for Task A."""
        specs = []

        scenarios = [
            ("baseline", []),
            ("proprio_noise_mild", [
                _ScheduledEvent(30, PerturbationConfig(
                    sensor_noise={"proprioception": 0.05},
                    label="proprio_noise_mild",
                    onset_step=30,
                ))
            ]),
            ("proprio_noise_heavy", [
                _ScheduledEvent(30, PerturbationConfig(
                    sensor_noise={"proprioception": 0.2},
                    label="proprio_noise_heavy",
                    onset_step=30,
                ))
            ]),
            ("vision_noise", [
                _ScheduledEvent(30, PerturbationConfig(
                    sensor_noise={"vision": 0.1},
                    label="vision_noise",
                    onset_step=30,
                ))
            ]),
            ("proprio_blackout_100", [
                _ScheduledEvent(50, PerturbationConfig(
                    sensor_blackout={"proprioception": 100},
                    label="proprio_blackout",
                    onset_step=50,
                ))
            ]),
            ("vision_blackout_80", [
                _ScheduledEvent(50, PerturbationConfig(
                    sensor_blackout={"vision": 80},
                    label="vision_blackout",
                    onset_step=50,
                ))
            ]),
            ("combined_noise_blackout", [
                _ScheduledEvent(50, PerturbationConfig(
                    sensor_noise={"proprioception": 0.08},
                    sensor_blackout={"vision": 80},
                    label="combined",
                    onset_step=50,
                ))
            ]),
            ("latency_100ms", [
                _ScheduledEvent(40, PerturbationConfig(
                    latency_steps=5,    # 5 × 20ms = 100ms
                    label="latency_100ms",
                    onset_step=40,
                ))
            ]),
        ]

        for label, schedule in scenarios:
            env = PerturbationHarness(
                MockReachEnv(max_steps=cls.MAX_STEPS, seed=seed),
                schedule=schedule,
                seed=seed,
            )
            specs.append(TaskSpec(
                name=f"reach/{label}",
                env=env,
                max_steps=cls.MAX_STEPS,
                success_fn=_reach_success,
                task_id="A",
                perturbation_label=label,
            ))
        return specs


class PerturbationSuite:
    """Combined Task A + C perturbation suite."""

    @classmethod
    def all_scenarios(cls, seed: Optional[int] = None) -> list[TaskSpec]:
        return ReachTask.scenarios(seed=seed) + BalanceTask.scenarios(seed=seed)

    @classmethod
    def task_a_only(cls, seed: Optional[int] = None) -> list[TaskSpec]:
        return ReachTask.scenarios(seed=seed)

    @classmethod
    def task_c_only(cls, seed: Optional[int] = None) -> list[TaskSpec]:
        return BalanceTask.scenarios(seed=seed)
