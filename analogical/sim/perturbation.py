"""Perturbation harness — wraps any BaseEnv and injects perturbations at runtime.

Perturbation types
------------------
actuator_dropout   : zero out actuator signal with probability p_drop
actuator_noise     : additive Gaussian noise on control signal (sigma)
sensor_noise       : additive Gaussian noise on per-modality observations
sensor_blackout    : completely zero a modality for blackout_steps steps
latency_injection  : delay observation delivery (simulated via cached obs)
force_kick         : apply random impulsive force to the underlying dynamics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .base import BaseEnv, SensoryBundle


@dataclass
class PerturbationConfig:
    """Specifies one perturbation event or ongoing profile."""

    # Actuator perturbations (Task C primary)
    actuator_dropout_prob: float = 0.0       # probability action → 0 each step
    actuator_noise_sigma: float = 0.0        # std of additive noise on action

    # Sensor perturbations (Task A primary)
    sensor_noise: Dict[str, float] = field(default_factory=dict)
    # e.g. {"proprioception": 0.05, "vision": 0.02}

    # Sensor blackout: modality → remaining_steps
    sensor_blackout: Dict[str, int] = field(default_factory=dict)
    # e.g. {"vision": 30}  — blacks out vision for 30 steps

    # Latency injection (affects which cached obs is returned)
    latency_steps: int = 0   # how many steps to delay observation

    # Force kick applied to the environment at the first step of a config
    force_kick_magnitude: float = 0.0  # applied to first action on that step

    # Metadata for logging
    label: str = ""
    onset_step: int = 0  # step at which perturbation begins (informational)


@dataclass
class _ScheduledEvent:
    onset_step: int
    config: PerturbationConfig


class PerturbationHarness(BaseEnv):
    """Wraps any BaseEnv and applies time-scheduled perturbation configs.

    Usage::

        harness = PerturbationHarness(MockCartpoleEnv(), schedule=[
            _ScheduledEvent(100, PerturbationConfig(actuator_dropout_prob=0.5)),
            _ScheduledEvent(200, PerturbationConfig(actuator_noise_sigma=1.0)),
        ])
        obs = harness.reset()
        for _ in range(500):
            action = agent.act(obs)
            obs, r, done, info = harness.step(action)
    """

    def __init__(
        self,
        env: BaseEnv,
        schedule: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._env = env
        self._schedule: list[_ScheduledEvent] = schedule or []
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._active_config = PerturbationConfig()
        self._obs_cache: list[SensoryBundle] = []   # for latency injection
        self._blackout_counters: Dict[str, int] = {}
        self._kick_pending = False

    # ── BaseEnv delegation ────────────────────────────────────────────────

    def reset(self) -> SensoryBundle:
        self._step_count = 0
        self._active_config = PerturbationConfig()
        self._obs_cache = []
        self._blackout_counters = {}
        self._kick_pending = False
        raw = self._env.reset()
        self._obs_cache = [raw]
        return raw

    def step(self, action: np.ndarray) -> tuple[SensoryBundle, float, bool, dict]:
        self._step_count += 1
        self._apply_schedule()

        cfg = self._active_config
        perturbed_action = self._perturb_action(action, cfg)

        raw_obs, reward, done, info = self._env.step(perturbed_action)

        # Update blackout counters
        for modality in list(self._blackout_counters.keys()):
            self._blackout_counters[modality] -= 1
            if self._blackout_counters[modality] <= 0:
                del self._blackout_counters[modality]
        # New blackouts from current config
        for modality, steps in cfg.sensor_blackout.items():
            if modality not in self._blackout_counters:
                self._blackout_counters[modality] = steps

        perturbed_obs = self._perturb_obs(raw_obs, cfg)

        # Latency buffer
        self._obs_cache.append(perturbed_obs)
        delay = max(0, cfg.latency_steps)
        delivered_obs = self._obs_cache[-delay - 1] if delay < len(self._obs_cache) else self._obs_cache[0]

        info["perturbation"] = cfg.label
        info["active_blackouts"] = list(self._blackout_counters.keys())
        return delivered_obs, reward, done, info

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._env.action_bounds

    @property
    def proprioception_dim(self) -> int:
        return self._env.proprioception_dim

    @property
    def task_name(self) -> str:
        return self._env.task_name

    @property
    def inner_env(self) -> BaseEnv:
        return self._env

    # ── Scheduling ────────────────────────────────────────────────────────

    def _apply_schedule(self) -> None:
        """Activate the latest scheduled event whose onset_step has passed."""
        for event in self._schedule:
            if event.onset_step <= self._step_count:
                self._active_config = event.config
            # Events are expected to be sorted by onset_step ascending

    # ── Perturbation application ──────────────────────────────────────────

    def _perturb_action(
        self, action: np.ndarray, cfg: PerturbationConfig
    ) -> np.ndarray:
        action = action.copy()

        # Actuator dropout
        if cfg.actuator_dropout_prob > 0:
            mask = self._rng.random(size=action.shape) > cfg.actuator_dropout_prob
            action *= mask.astype(action.dtype)

        # Actuator noise
        if cfg.actuator_noise_sigma > 0:
            action = action + self._rng.normal(
                0, cfg.actuator_noise_sigma, size=action.shape
            ).astype(action.dtype)
            low, high = self._env.action_bounds
            action = np.clip(action, low, high)

        # Force kick (adds impulse on top of action, first step only)
        if cfg.force_kick_magnitude > 0 and not self._kick_pending:
            self._kick_pending = True
            kick = self._rng.uniform(-1, 1, size=action.shape)
            kick = kick / (np.linalg.norm(kick) + 1e-8) * cfg.force_kick_magnitude
            action = action + kick.astype(action.dtype)
            low, high = self._env.action_bounds
            action = np.clip(action, low, high)

        return action

    def _perturb_obs(
        self, obs: SensoryBundle, cfg: PerturbationConfig
    ) -> SensoryBundle:
        obs = obs.clone()

        # Sensor noise
        for modality, sigma in cfg.sensor_noise.items():
            arr = getattr(obs, modality, None)
            if arr is not None and sigma > 0:
                noisy = arr + self._rng.normal(0, sigma, size=arr.shape).astype(arr.dtype)
                setattr(obs, modality, noisy)

        # Sensor blackout (active counters)
        for modality in self._blackout_counters:
            setattr(obs, modality, None)

        return obs

    # ── Factory helpers ───────────────────────────────────────────────────

    @classmethod
    def make_task_c_suite(
        cls, env: BaseEnv, seed: Optional[int] = None
    ) -> list["PerturbationHarness"]:
        """Return standard Task-C perturbation scenarios for balance task."""
        scenarios = [
            # Baseline — no perturbation
            [],
            # Mild actuator noise
            [_ScheduledEvent(50, PerturbationConfig(
                actuator_noise_sigma=0.3, label="mild_noise"
            ))],
            # Heavy dropout (30%)
            [_ScheduledEvent(50, PerturbationConfig(
                actuator_dropout_prob=0.3, label="dropout_30pct"
            ))],
            # Heavy dropout (60%)
            [_ScheduledEvent(50, PerturbationConfig(
                actuator_dropout_prob=0.6, label="dropout_60pct"
            ))],
            # Force kick + noise
            [_ScheduledEvent(80, PerturbationConfig(
                force_kick_magnitude=2.0, actuator_noise_sigma=0.5, label="kick_plus_noise"
            ))],
            # Escalating dropout
            [
                _ScheduledEvent(0, PerturbationConfig(label="clean")),
                _ScheduledEvent(100, PerturbationConfig(actuator_dropout_prob=0.2, label="ramp_20pct")),
                _ScheduledEvent(200, PerturbationConfig(actuator_dropout_prob=0.4, label="ramp_40pct")),
                _ScheduledEvent(300, PerturbationConfig(actuator_dropout_prob=0.6, label="ramp_60pct")),
            ],
        ]
        return [cls(env, schedule=s, seed=seed) for s in scenarios]

    @classmethod
    def make_task_a_suite(
        cls, env: BaseEnv, seed: Optional[int] = None
    ) -> list["PerturbationHarness"]:
        """Return standard Task-A perturbation scenarios for reaching task."""
        scenarios = [
            # Baseline
            [],
            # Proprioception noise
            [_ScheduledEvent(30, PerturbationConfig(
                sensor_noise={"proprioception": 0.05}, label="proprio_noise"
            ))],
            # Vision noise
            [_ScheduledEvent(30, PerturbationConfig(
                sensor_noise={"vision": 0.1}, label="vision_noise"
            ))],
            # Proprioception blackout (100 steps)
            [_ScheduledEvent(50, PerturbationConfig(
                sensor_blackout={"proprioception": 100}, label="proprio_blackout"
            ))],
            # Vision blackout
            [_ScheduledEvent(50, PerturbationConfig(
                sensor_blackout={"vision": 100}, label="vision_blackout"
            ))],
            # Combined: proprio noise + vision blackout
            [_ScheduledEvent(50, PerturbationConfig(
                sensor_noise={"proprioception": 0.08},
                sensor_blackout={"vision": 80},
                label="combined"
            ))],
            # Latency injection (5 steps ≈ 100ms at 20Hz)
            [_ScheduledEvent(40, PerturbationConfig(
                latency_steps=5, label="latency_100ms"
            ))],
        ]
        return [cls(env, schedule=s, seed=seed) for s in scenarios]
