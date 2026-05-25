"""Online adaptation: modality reliability routing + Hebbian plastic updates.

ModalityRouter
--------------
Tracks per-modality reliability via an exponential moving average (EMA) of
prediction consistency.  Reliability = how well the world model can predict
the next latent state when the modality is present.  The router outputs
soft attention weights so high-reliability modalities contribute more to the
shared latent state — without any explicit mode switching.

PlasticUpdater
--------------
Applies small Hebbian-inspired gradient updates to LatentCore parameters
after each step, allowing the network to continuously re-specialise its
internal representations to the current sensory regime.  A decaying learning
rate schedule prevents catastrophic forgetting of early-episode structure.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from analogical.perception.latent_core import LatentCore
from analogical.control.world_model import OnlineWorldModel


class ModalityRouter:
    """Reliability-weighted soft routing across sensory modalities.

    Uses prediction-error feedback from the world model: if the world model
    predicts the next latent well when modality *m* is present, *m* is deemed
    reliable.  When *m* is absent or noisy, its reliability decays and other
    modalities automatically compensate.

    Parameters
    ----------
    ema_alpha : float
        Smoothing factor for the reliability EMA (0 < α < 1).
        Smaller → slower, more stable; larger → faster adaptation.
    """

    def __init__(self, ema_alpha: float = 0.15) -> None:
        self.ema_alpha = ema_alpha
        self._reliability: Dict[str, float] = {}
        self._presence_history: Dict[str, List[bool]] = {}
        self._history_len = 20   # steps kept for recent-presence check

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        present_modalities: List[str],
        all_modalities: List[str],
        prediction_error: float,
    ) -> None:
        """Update reliability scores after a step.

        Parameters
        ----------
        present_modalities : list of modality names that were non-None this step
        all_modalities     : all modality names the system knows about
        prediction_error   : world-model 1-step MSE loss (lower = world model
                             predicts well given current sensing)
        """
        # Convert prediction error to a quality signal in [0, 1]
        quality = float(1.0 / (1.0 + prediction_error))

        for m in all_modalities:
            present = m in present_modalities
            # Track presence history
            hist = self._presence_history.setdefault(m, [])
            hist.append(present)
            if len(hist) > self._history_len:
                hist.pop(0)

            # Update reliability EMA
            # If modality is present: use quality (world model accuracy)
            # If modality is absent : reliability decays
            current = self._reliability.get(m, 0.5)
            if present:
                target = quality
            else:
                target = 0.0
            self._reliability[m] = (
                self.ema_alpha * target + (1.0 - self.ema_alpha) * current
            )

    def get_reliability(self, modality: str) -> float:
        """Return current reliability score in [0, 1]."""
        return self._reliability.get(modality, 0.5)

    def routing_weights(self, modalities: List[str]) -> Dict[str, float]:
        """Softmax-normalised reliability weights for the given modality set."""
        if not modalities:
            return {}
        scores = {m: self._reliability.get(m, 0.5) for m in modalities}
        total = sum(scores.values())
        if total < 1e-8:
            n = len(modalities)
            return {m: 1.0 / n for m in modalities}
        return {m: v / total for m, v in scores.items()}

    def reset_episode(self) -> None:
        """Reset reliability to neutral at episode start (but keep EMA)."""
        # Deliberately NOT clearing — reliability should persist across episodes
        # to retain adaptation. Reset is only for per-episode counters.
        self._presence_history = {}


class PlasticUpdater:
    """Hebbian-inspired online plastic updates to LatentCore parameters.

    Applies a small gradient step on the core's output consistency loss every
    N_update steps.  Uses a decaying learning rate to prevent drift.

    The update rule targets self-consistency: the latent state at step t
    should be predictable from step t-1.  This is computed from the world
    model's prediction error without any external labels.

    Parameters
    ----------
    core      : LatentCore to be updated
    base_lr   : peak learning rate for plastic updates
    decay     : multiplicative decay per step (learning rate schedule)
    update_every : apply one gradient step every N steps
    max_norm  : gradient clipping norm
    """

    def __init__(
        self,
        core: LatentCore,
        base_lr: float = 1e-4,
        decay: float = 0.9999,
        update_every: int = 5,
        max_norm: float = 0.5,
    ) -> None:
        self._core = core
        self._base_lr = base_lr
        self._current_lr = base_lr
        self._decay = decay
        self._update_every = update_every
        self._max_norm = max_norm
        self._step = 0

        self._optimizer = torch.optim.Adam(core.parameters(), lr=base_lr)

    # ── Update step ────────────────────────────────────────────────────────

    def maybe_update(
        self,
        latent_t: torch.Tensor,
        tokens_next: dict,
        reliability_next: dict,
        world_model: OnlineWorldModel,
        action: torch.Tensor,
    ) -> Optional[float]:
        """Optionally run one plastic update step.

        Parameters
        ----------
        latent_t         : latent state at current step (detached)
        tokens_next      : encoded modality tokens for the next step
        reliability_next : reliability weights for the next step
        world_model      : used to get the prediction target (no grad needed)
        action           : action taken this step

        Returns
        -------
        loss value if update was applied, else None
        """
        self._step += 1
        self._current_lr = self._base_lr * (self._decay ** self._step)

        if self._step % self._update_every != 0:
            return None

        # Detached prediction target from world model
        with torch.no_grad():
            pred_next, _, _ = world_model(latent_t, action)

        # Recompute next latent through core WITH gradients so that
        # core parameters accumulate meaningful gradient signal.
        with torch.enable_grad():
            latent_next = self._core(tokens_next, reliability_next, reset_hidden=False)
            # Cosine similarity loss — encourages alignment without magnitude collapse
            loss = 1.0 - torch.nn.functional.cosine_similarity(
                pred_next.detach().unsqueeze(0), latent_next.unsqueeze(0)
            ).mean()

            # Update learning rate for optimizer
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._current_lr

            self._optimizer.zero_grad()
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self._core.parameters(), self._max_norm)
            self._optimizer.step()

        return float(loss.item())

    def reset_episode(self) -> None:
        self._step = 0
        self._current_lr = self._base_lr
