"""Continuous Hebbian learning for LatentCore's recurrent representations.

Motivation
----------
Gradient-based learning (backprop) operates in discrete offline batches.
Biological learning is *continuous* — synaptic weights adjust after every
spike.  This module implements the **Oja rule**, a normalised Hebbian update:

    ΔW_hh = η · (h_new ⊗ h_old − (h_new · h_new) · W_hh · h_old ⊗ h_old)

The Oja correction term prevents weight blow-up (it acts as a self-normalising
constraint), keeping the recurrent weights bounded without explicit clipping.

The update is applied to the GRU cell's recurrent weight matrix (``weight_hh``)
inside LatentCore every step.  Combined with the gradient-based PlasticUpdater,
this gives the system two interacting learning timescales:

  - Hebbian (this module): every step, very small η — fast, local, continuous
  - Gradient (PlasticUpdater): every N steps, larger LR — slower, global

Together they mirror the dual-process view of biological learning
(fast Hebbian consolidation + slower error-correcting STDP/backprop-like rules).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from analogical.perception.latent_core import LatentCore


class HebbianUpdater:
    """Applies the Oja Hebbian update rule to LatentCore's GRU every step.

    Parameters
    ----------
    core         : LatentCore whose GRU recurrent weights are to be updated
    lr           : Hebbian learning rate η (default 1e-5 — deliberately small)
    decay        : additional L2 weight decay per step (prevents slow drift)
    update_every : apply once per N steps (1 = every step; raise for CPU speed)
    """

    def __init__(
        self,
        core: LatentCore,
        lr: float = 1e-5,
        decay: float = 1e-6,
        update_every: int = 1,
    ) -> None:
        self._core = core
        self.lr = lr
        self.decay = decay
        self.update_every = update_every
        self._step = 0

        # Grab reference to the GRU's recurrent weight parameter
        # GRUCell stores weights as weight_hh of shape (3*hidden, hidden)
        self._gru = core.gru
        if not hasattr(self._gru, "weight_hh"):
            raise AttributeError(
                "LatentCore.gru must be an nn.GRUCell with a 'weight_hh' attribute."
            )

    # ── Update ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def update(self, h_old: torch.Tensor, h_new: torch.Tensor) -> None:
        """Apply one Oja-rule Hebbian step.

        Parameters
        ----------
        h_old : GRU hidden state before the current step  (hidden_dim,)
        h_new : GRU hidden state after  the current step  (hidden_dim,)
        """
        self._step += 1
        if self._step % self.update_every != 0:
            return

        W = self._gru.weight_hh.data  # (3*hidden, hidden)
        d = h_old.shape[0]
        W_r = W[d : 2 * d]  # reset-gate slice

        # Normalise inputs so update magnitude is bounded regardless of hidden scale
        eps = 1e-8
        h_old_n = h_old / (h_old.norm() + eps)  # unit norm  (hidden,)
        h_new_n = h_new / (h_new.norm() + eps)  # unit norm  (hidden,)

        # Hebbian outer product
        outer = h_new_n.unsqueeze(1) * h_old_n.unsqueeze(0)  # (hidden, hidden)

        # Oja correction: remove the component already captured by current W.
        # y = W_r @ h_old_n  is the "reconstruction" of h_new via current weights.
        # Subtracting h_new_n ⊗ (y * h_old_n) prevents runaway potentiation.
        y = W_r @ h_old_n                                          # (hidden,)
        y_times_hold = y * h_old_n                                 # (hidden,)
        correction = h_new_n.unsqueeze(1) * y_times_hold.unsqueeze(0)  # (hidden, hidden)

        delta = self.lr * (outer - correction) - self.decay * W_r
        W[d : 2 * d].add_(delta)

        # Hard clamp: safety net for extreme lr values or random initialisation
        W[d : 2 * d].clamp_(-3.0, 3.0)

    # ── Episode management ─────────────────────────────────────────────────

    def reset_episode(self) -> None:
        self._step = 0
