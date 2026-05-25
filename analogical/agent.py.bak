"""AnalogicalAgent — the full perception-first, zero-pretraining adaptive agent.

Execution flow per step
-----------------------
1. SensoryBundle → per-modality encoders → reliability scores
2. ModalityRouter updates routing weights from previous prediction error
3. LatentCore fuses weighted tokens into latent state z_t (with GRU history)
4. MPPIPlanner rolls out OnlineWorldModel under budget from ComputeGovernor
5. PlasticUpdater optionally updates LatentCore for online specialisation
6. OnlineWorldModel.observe() stores transition and runs gradient step
7. Action returned; timing logged to ComputeGovernor

Design invariants
-----------------
- No pretrained weights. All networks start with Kaiming random init.
- 200 ms hard cap on act() enforced by MPPI planner + governor.
- If all sensors drop out, agent falls back to world-model hallucinated latent.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from analogical.governor.compute import ComputeGovernor
from analogical.perception.encoders import (
    AudioEncoder,
    ProprioceptionEncoder,
    TactileEncoder,
    VisionEncoder,
)
from analogical.perception.latent_core import LatentCore
from analogical.control.world_model import OnlineWorldModel
from analogical.control.planner import MPPIPlanner, HARD_CAP_MS
from analogical.adaptation.routing import ModalityRouter, PlasticUpdater
from analogical.sim.base import SensoryBundle


class AnalogicalAgent:
    """Modality-agnostic real-time adaptive agent.

    Parameters
    ----------
    proprioception_dim : int  — must match the environment
    action_dim         : int  — must match the environment
    action_low         : np.ndarray (action_dim,)
    action_high        : np.ndarray (action_dim,)
    d_model            : int  — shared latent dimensionality (default 256)
    n_transformer_heads: int
    n_transformer_layers: int
    mppi_n_samples     : int  — initial MPPI sample count (adaptive)
    mppi_horizon       : int  — initial MPPI horizon (adaptive)
    device             : torch.device or None (auto-select)
    """

    def __init__(
        self,
        proprioception_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        d_model: int = 256,
        n_transformer_heads: int = 4,
        n_transformer_layers: int = 2,
        mppi_n_samples: int = 512,
        mppi_horizon: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        # Compute governor first — determines device
        self.governor = ComputeGovernor(
            hard_cap_ms=HARD_CAP_MS,
            prefer_gpu=True,
        )
        self._device = device or self.governor.device

        # ── Encoders (one per expected modality) ──────────────────────────
        self.encoders = torch.nn.ModuleDict({
            "proprioception": ProprioceptionEncoder(proprioception_dim, d_model),
            "vision": VisionEncoder(d_model),
            "tactile": TactileEncoder(1, d_model),
            "audio": AudioEncoder(d_model=d_model),
        })
        for enc in self.encoders.values():
            enc.to(self._device)

        # ── Latent core ───────────────────────────────────────────────────
        self.core = LatentCore(
            d_model=d_model,
            n_heads=n_transformer_heads,
            n_layers=n_transformer_layers,
        ).to(self._device)

        # ── World model ───────────────────────────────────────────────────
        self.world_model = OnlineWorldModel(
            latent_dim=d_model,
            action_dim=action_dim,
        ).to(self._device)

        # ── Planner ───────────────────────────────────────────────────────
        self.planner = MPPIPlanner(
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            n_samples=mppi_n_samples,
            horizon=mppi_horizon,
            device=self._device,
        )

        # ── Adaptation ────────────────────────────────────────────────────
        self.router = ModalityRouter(ema_alpha=0.15)
        self.plastic = PlasticUpdater(self.core, base_lr=1e-4, update_every=5)

        # ── Episode state ─────────────────────────────────────────────────
        self._last_latent: Optional[torch.Tensor] = None
        self._last_action: Optional[np.ndarray] = None
        self._step: int = 0
        self._last_wm_loss: float = 0.0

        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.d_model = d_model

    # ── Episode reset ─────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        self.core.reset_episode()
        self.world_model.reset_episode()
        self.planner.reset_episode()
        self.router.reset_episode()
        self.plastic.reset_episode()
        self._last_latent = None
        self._last_action = None
        self._step = 0
        self._last_wm_loss = 0.0

    # ── Primary act interface ─────────────────────────────────────────────

    def act(
        self,
        bundle: SensoryBundle,
    ) -> tuple[np.ndarray, float, dict]:
        """Choose an action from a SensoryBundle.

        Returns
        -------
        action      : np.ndarray (action_dim,)
        elapsed_ms  : float — total act() wall-clock time
        debug       : dict
        """
        t0 = time.monotonic()

        with torch.no_grad():
            # 1. Encode each available modality
            tokens, raw_reliability = self._encode(bundle)

            # 2. Routing weights from router (updated by previous loss)
            all_modalities = list(self.encoders.keys())
            present = list(tokens.keys())
            routing = self.router.routing_weights(present)

            # Merge: use router weights, fall back to raw availability
            reliability_for_core = {
                m: routing.get(m, raw_reliability.get(m, 0.0)) * raw_reliability.get(m, 0.0)
                for m in present
            }

            # 3. Fuse in latent core
            reset = (self._step == 0)
            latent = self.core(tokens, reliability_for_core, reset_hidden=reset)
            latent = latent.to(self._device)

            # 4. Plan with MPPI under governor budget
            n = self.governor.get_n_samples()
            h = self.governor.get_horizon()

            action_arr, plan_ms, plan_info = self.planner.plan(
                latent,
                self.world_model,
                deadline_ms=HARD_CAP_MS,
                n_samples_override=n,
                horizon_override=h,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self.governor.record_step_time(elapsed_ms)

        self._last_latent = latent
        self._last_action = action_arr
        self._step += 1

        return action_arr, elapsed_ms, {
            "plan_ms": round(plan_ms, 2),
            "deadline_met": elapsed_ms <= HARD_CAP_MS,
            "budget_level": self.governor.current_level.name,
            **plan_info,
        }

    # ── Observe (online update after environment step) ─────────────────────

    def observe(
        self,
        prev_bundle: SensoryBundle,
        action: np.ndarray,
        next_bundle: SensoryBundle,
        reward: float,
    ) -> Optional[float]:
        """Store transition and run online updates.

        Returns world model loss (or None if buffer not warm yet).
        """
        if self._last_latent is None:
            return None

        with torch.no_grad():
            tokens_next, _ = self._encode(next_bundle)
            present_next = list(tokens_next.keys())
            routing_next = self.router.routing_weights(present_next)
            latent_next = self.core(tokens_next, routing_next, reset_hidden=False)
            latent_next = latent_next.to(self._device)

        action_t = torch.tensor(action, dtype=torch.float32, device=self._device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self._device)
        latent_t = self._last_latent.to(self._device)

        # World model online update
        wm_loss = self.world_model.observe(latent_t, action_t, latent_next, reward_t)
        self._last_wm_loss = wm_loss

        # Router update: use world model accuracy as reliability signal
        all_m = list(self.encoders.keys())
        present_m = list(tokens_next.keys())
        self.router.update(present_m, all_m, prediction_error=wm_loss)

        # Plastic update to LatentCore (pass tokens so it can recompute with grad)
        plastic_loss = self.plastic.maybe_update(
            latent_t=latent_t,
            tokens_next=tokens_next,
            reliability_next=routing_next,
            world_model=self.world_model,
            action=action_t,
        )

        return wm_loss

    # ── Internal helpers ───────────────────────────────────────────────────

    def _encode(
        self, bundle: SensoryBundle
    ) -> tuple[dict, dict]:
        """Encode all available modalities. Returns (tokens, raw_reliability)."""
        tokens: dict = {}
        raw_reliability: dict = {}

        modality_data = {
            "proprioception": bundle.proprioception,
            "vision": bundle.vision,
            "audio": bundle.audio,
            "tactile": bundle.tactile,
        }

        for name, arr in modality_data.items():
            if arr is None:
                raw_reliability[name] = 0.0
                continue
            t = torch.tensor(arr, dtype=torch.float32, device=self._device)
            enc = self.encoders[name]
            emb, rel = enc(t)
            tokens[name] = emb.to(self._device)
            raw_reliability[name] = float(rel.item())

        return tokens, raw_reliability

    # ── Convenience factory ────────────────────────────────────────────────

    @classmethod
    def from_env(cls, env, **kwargs) -> "AnalogicalAgent":
        """Construct agent from a BaseEnv instance."""
        low, high = env.action_bounds
        return cls(
            proprioception_dim=env.proprioception_dim,
            action_dim=env.action_dim,
            action_low=low,
            action_high=high,
            **kwargs,
        )
