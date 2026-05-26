"""AnalogicalAgent — perception-first, zero-pretraining adaptive agent.

Learning systems
----------------
1. **Gradient-based world model** (OnlineWorldModel)
   Mini-batch gradient step every step from a sliding buffer of recent
   transitions.  Learns environment dynamics online.

2. **Gradient-based plastic updates** (PlasticUpdater)
   Every N steps: cosine-alignment loss through LatentCore so that its
   representations stay consistent with world-model predictions.

3. **Continuous Hebbian learning** (HebbianUpdater)
   Every single step, no batching: Oja rule applied to the GRU recurrent
   weights inside LatentCore.  Fast, local, biologically-plausible.
   Mirrors synaptic potentiation — frequently co-active hidden units form
   stronger connections.

4. **Neural memory bank** (NeuralMemoryBank)
   Associative memory of (latent_context, action, return) triples.
   Retrieval attention = cosine_sim x log(1 + frequency) / tau.
   The top recalled action seeds the MPPI warm-start: frequently-used
   successful actions are recalled more strongly and bias the planner
   prior, just as procedural memory biases human motor control.
   Consolidation: top-K high-frequency memories are periodically replayed
   through the world model to strengthen their dynamic traces.

5. **Modality routing** (ModalityRouter)
   EMA reliability scores -> soft weighting in LatentCore fusion.

Execution flow per act() call
------------------------------
  SensoryBundle
    -> per-modality encoders
    -> ModalityRouter weights
    -> LatentCore (transformer + GRU)  <- Hebbian update applied here
    -> NeuralMemoryBank.query()        <- frequency-weighted action prior
    -> MPPIPlanner (warm-started with memory prior)
    -> action, elapsed_ms

Execution flow per observe() call
----------------------------------
  (prev_obs, action, next_obs, reward)
    -> encode next obs -> next latent
    -> OnlineWorldModel.observe()      <- gradient step
    -> NeuralMemoryBank.write()        <- store / update memory slot
    -> ModalityRouter.update()         <- update reliability EMA
    -> PlasticUpdater.maybe_update()   <- gradient step on core
    -> NeuralMemoryBank.consolidate()  <- replay top-K into world model
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from analogical.adaptation.hebbian import HebbianUpdater
from analogical.adaptation.routing import ModalityRouter, PlasticUpdater
from analogical.control.planner import HARD_CAP_MS, MPPIPlanner
from analogical.control.world_model import OnlineWorldModel
from analogical.governor.compute import ComputeGovernor
from analogical.memory.neural_memory import NeuralMemoryBank
from analogical.perception.encoders import (
    AudioEncoder,
    ProprioceptionEncoder,
    TactileEncoder,
    VisionEncoder,
)
from analogical.perception.latent_core import LatentCore
from analogical.sim.base import SensoryBundle

# How often to run memory consolidation replay (steps).
# 20 steps ≈ 1 second at 20 Hz — frequent enough to strengthen emerging
# memories without consuming significant per-step budget.
_CONSOLIDATE_EVERY: int = 20

# Minimum memory confidence required before overriding the MPPI warm-start.
# Below this level the memory is too uncertain to be a useful prior; MPPI
# uses its own (shifted) warm-start instead.
_MEMORY_CONFIDENCE_THRESHOLD: float = 0.2


class AnalogicalAgent:
    """Modality-agnostic real-time adaptive agent with continuous online learning.

    Parameters
    ----------
    proprioception_dim  : int
    action_dim          : int
    action_low          : np.ndarray (action_dim,)
    action_high         : np.ndarray (action_dim,)
    d_model             : int (default 256)
    n_transformer_heads : int
    n_transformer_layers: int
    mppi_n_samples      : int
    mppi_horizon        : int
    memory_capacity     : int — max slots in NeuralMemoryBank
    device              : torch.device or None (auto-select)
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
        memory_capacity: int = 1024,
        device: Optional[torch.device] = None,
    ) -> None:
        self.governor = ComputeGovernor(hard_cap_ms=HARD_CAP_MS, prefer_gpu=True)
        self._device = device or self.governor.device

        # ── Encoders ──────────────────────────────────────────────────────
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

        # ── Continuous Hebbian learning ───────────────────────────────────
        self.hebbian = HebbianUpdater(self.core, lr=1e-5, decay=1e-6, update_every=1)

        # ── Neural memory bank ────────────────────────────────────────────
        self.memory = NeuralMemoryBank(
            capacity=memory_capacity,
            d_model=d_model,
            action_dim=action_dim,
            temperature=0.1,
        )

        # ── Modality routing + gradient plastic update ────────────────────
        self.router = ModalityRouter(ema_alpha=0.15)
        self.plastic = PlasticUpdater(self.core, base_lr=1e-4, update_every=5)

        # ── Episode state ─────────────────────────────────────────────────
        self._last_latent: Optional[torch.Tensor] = None
        self._last_action: Optional[np.ndarray] = None
        self._step: int = 0
        self._episode_return: float = 0.0

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
        self.hebbian.reset_episode()
        self.memory.reset_episode()
        self._last_latent = None
        self._last_action = None
        self._step = 0
        self._episode_return = 0.0

    # ── Primary act interface ─────────────────────────────────────────────

    def act(
        self,
        bundle: SensoryBundle,
    ) -> tuple[np.ndarray, float, dict]:
        """Choose an action from a SensoryBundle.

        Returns
        -------
        action      : np.ndarray (action_dim,)
        elapsed_ms  : float
        debug       : dict
        """
        t0 = time.monotonic()

        with torch.no_grad():
            # 1. Encode
            tokens, raw_reliability = self._encode(bundle)

            # 2. Routing weights
            present = list(tokens.keys())
            routing = self.router.routing_weights(present)
            reliability_for_core = {
                m: routing.get(m, raw_reliability.get(m, 0.0)) * raw_reliability.get(m, 0.0)
                for m in present
            }

            # 3. Capture GRU hidden state BEFORE this step (for Hebbian)
            h_old = (
                self.core.hidden_state.clone()
                if self.core.hidden_state is not None
                else torch.zeros(self.d_model, device=self._device)
            )

            # 4. Fuse in latent core
            reset = (self._step == 0)
            latent = self.core(tokens, reliability_for_core, reset_hidden=reset)
            latent = latent.to(self._device)

            # 5. Continuous Hebbian update on recurrent weights
            h_new = self.core.hidden_state
            if h_new is not None:
                self.hebbian.update(h_old.cpu(), h_new.cpu())

            # 6. Query neural memory for action prior
            action_prior, mem_value, mem_confidence = self.memory.query(latent.cpu())

            # 7. Warm-start MPPI with memory prior when confidence is meaningful
            if action_prior is not None and mem_confidence > _MEMORY_CONFIDENCE_THRESHOLD:
                prior_arr = np.clip(action_prior.numpy(), self.action_low, self.action_high)
                h = self.governor.get_horizon()
                prior_t = torch.tensor(prior_arr, dtype=torch.float32, device=self._device)
                self.planner._prev_seq = prior_t.unsqueeze(0).expand(h, -1).clone()

            # 8. Plan
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
            "mem_confidence": round(mem_confidence, 3),
            "mem_size": self.memory._size,
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
        """Store transition and run all online update systems."""
        if self._last_latent is None:
            return None

        self._episode_return += reward

        with torch.no_grad():
            tokens_next, _ = self._encode(next_bundle)
            present_next = list(tokens_next.keys())
            routing_next = self.router.routing_weights(present_next)
            latent_next = self.core(tokens_next, routing_next, reset_hidden=False)
            latent_next = latent_next.to(self._device)

        action_t = torch.tensor(action, dtype=torch.float32, device=self._device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self._device)
        latent_t = self._last_latent.to(self._device)

        # World model gradient update
        wm_loss = self.world_model.observe(latent_t, action_t, latent_next, reward_t)

        # Neural memory write (value = per-step reward)
        self.memory.write(latent_t.cpu(), action_t.cpu(), reward)

        # Modality router update
        all_m = list(self.encoders.keys())
        self.router.update(list(tokens_next.keys()), all_m, prediction_error=wm_loss)

        # Gradient plastic update on LatentCore
        self.plastic.maybe_update(
            latent_t=latent_t,
            tokens_next=tokens_next,
            reliability_next=routing_next,
            world_model=self.world_model,
            action=action_t,
        )

        # Memory consolidation replay (every N steps)
        if self._step % _CONSOLIDATE_EVERY == 0:
            self._consolidate()

        return wm_loss

    # ── Internal helpers ───────────────────────────────────────────────────

    def _encode(
        self, bundle: SensoryBundle
    ) -> tuple[dict, dict]:
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
            emb, rel = self.encoders[name](t)
            tokens[name] = emb.to(self._device)
            raw_reliability[name] = float(rel.item())

        return tokens, raw_reliability

    def _consolidate(self) -> None:
        """Replay top-K high-frequency memories through the world model."""
        for key, action, value, _ in self.memory.top_k_memories():
            latent_t = key.to(self._device)
            action_t = action.to(self._device)
            with torch.no_grad():
                pred_next, _, _ = self.world_model(latent_t, action_t)
            reward_t = torch.tensor([value], dtype=torch.float32, device=self._device)
            self.world_model.observe(latent_t, action_t, pred_next, reward_t)

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
