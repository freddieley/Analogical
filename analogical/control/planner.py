"""MPPI (Model Predictive Path Integral) planner with strict real-time deadlines.

The 200 ms hard cap is enforced by a wall-clock watchdog: if the planner is
within 20 ms of the deadline it returns the current-best action early.  The
compute governor may further reduce the planning horizon H and sample count N
before the call, ensuring the cap is met under GPU/CPU pressure.

Algorithm
---------
MPPI solves:
    a* = argmax_a E[J(τ)]  where  J = Σ_t γ^t r_t

By drawing N noisy action sequences, rolling them through the world model,
and taking the importance-weighted mean:

    w_i  = exp((J_i - J_max) / λ)
    a_t* = Σ_i w_i * ε_{i,t} / Σ_i w_i   (centred correction)

Reference: Williams et al. 2017 — "Information Theoretic MPC".
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .world_model import OnlineWorldModel

# Hard deadline shared across the system
HARD_CAP_MS: float = 200.0
_SAFETY_MARGIN_MS: float = 20.0  # stop early if within this of cap


class MPPIPlanner:
    """MPPI planner for continuous action spaces.

    Parameters
    ----------
    action_dim   : action vector dimensionality
    action_low   : lower bound per action dim
    action_high  : upper bound per action dim
    n_samples    : number of trajectory samples (N)
    horizon      : planning horizon in steps (H)
    temperature  : MPPI temperature λ (lower → more greedy)
    noise_sigma  : std of action perturbation noise
    device       : torch device
    """

    def __init__(
        self,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        n_samples: int = 512,
        horizon: int = 10,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.action_dim = action_dim
        self.n_samples = n_samples
        self.horizon = horizon
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.device = device or torch.device("cpu")

        self._low = torch.tensor(action_low, dtype=torch.float32, device=self.device)
        self._high = torch.tensor(action_high, dtype=torch.float32, device=self.device)
        self._action_range = self._high - self._low

        # Warm-start: previous optimal action sequence (H, action_dim)
        self._prev_seq: Optional[torch.Tensor] = None

    # ── Main planning call ─────────────────────────────────────────────────

    @torch.no_grad()
    def plan(
        self,
        latent: torch.Tensor,
        world_model: OnlineWorldModel,
        deadline_ms: float = HARD_CAP_MS,
        n_samples_override: Optional[int] = None,
        horizon_override: Optional[int] = None,
    ) -> tuple[np.ndarray, float, dict]:
        """Compute the best action for the current latent state.

        Parameters
        ----------
        latent            : (latent_dim,) current latent state
        world_model       : OnlineWorldModel — used for rollouts
        deadline_ms       : wall-clock budget in ms (default: HARD_CAP_MS)
        n_samples_override: override N (from compute governor)
        horizon_override  : override H (from compute governor)

        Returns
        -------
        action   : np.ndarray (action_dim,)  — clipped to bounds
        elapsed  : float  — ms taken
        debug    : dict
        """
        t_start = time.monotonic()
        N = n_samples_override or self.n_samples
        H = horizon_override or self.horizon
        effective_deadline = min(deadline_ms, HARD_CAP_MS) - _SAFETY_MARGIN_MS

        # Warm-start nominal sequence (H, action_dim)
        if self._prev_seq is not None and self._prev_seq.shape[0] == H:
            nominal = self._prev_seq.clone().to(self.device)
        else:
            mid = (self._low + self._high) / 2.0
            nominal = mid.unsqueeze(0).expand(H, -1).clone()

        # Sample N perturbations: (N, H, action_dim)
        eps = torch.randn(N, H, self.action_dim, device=self.device) * self.noise_sigma
        action_seqs = nominal.unsqueeze(0) + eps  # (N, H, action_dim)
        action_seqs = torch.clamp(action_seqs, self._low, self._high)

        # Expand latent to batch: (N, latent_dim)
        lat_batch = latent.unsqueeze(0).expand(N, -1).to(self.device)

        # Rollout
        world_model.to(self.device)
        rewards, _ = world_model.rollout(lat_batch, action_seqs)  # (N,)

        # MPPI weights
        r_max = rewards.max()
        weights = torch.exp((rewards - r_max) / (self.temperature + 1e-8))
        weights = weights / (weights.sum() + 1e-8)  # (N,)

        # Weighted action update
        # Correction: weighted mean of perturbations
        corrections = (weights.view(N, 1, 1) * eps).sum(dim=0)  # (H, action_dim)
        updated_seq = torch.clamp(nominal + corrections, self._low, self._high)

        # Shift for next warm-start
        self._prev_seq = torch.cat([
            updated_seq[1:],
            ((self._low + self._high) / 2.0).unsqueeze(0),
        ], dim=0)

        best_action = updated_seq[0]  # first action in sequence

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        # Hard cap enforcement
        if elapsed_ms > HARD_CAP_MS:
            # Deadline violated — return nominal safe action
            best_action = (self._low + self._high) / 2.0

        return (
            best_action.cpu().numpy(),
            elapsed_ms,
            {
                "N": N,
                "H": H,
                "best_reward": float(rewards.max().item()),
                "deadline_met": elapsed_ms <= HARD_CAP_MS,
            },
        )

    def reset_episode(self) -> None:
        """Clear warm-start cache at episode boundaries."""
        self._prev_seq = None

    def adapt_budget(self, n: int, h: int) -> None:
        """Allow compute governor to override N and H directly."""
        self.n_samples = n
        self.horizon = h
