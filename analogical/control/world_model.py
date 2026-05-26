"""Online world model — GRU-based latent dynamics + reward predictor.

Learns purely from the current episode's experience. No pretrained weights.
Updated every step via a sliding window of recent transitions.

Architecture
------------
  dynamics : GRU  (latent_t, action_t) → hidden → latent_{t+1}_pred
  reward   : MLP  latent_t → r_t_pred
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineWorldModel(nn.Module):
    """GRU world model updated online via 1-step prediction error.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent state from LatentCore (default 256).
    action_dim : int
        Action vector size.
    hidden_dim : int
        GRU hidden size.
    buffer_size : int
        Number of recent transitions kept for each online update.
    online_lr : float
        Adam learning rate for online updates.
    online_batch : int
        Mini-batch size sampled from buffer on each update.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 1,
        hidden_dim: int = 256,
        buffer_size: int = 64,
        online_lr: float = 5e-4,
        online_batch: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input projection: concat(latent, action) → hidden_dim
        self.input_proj = nn.Linear(latent_dim + action_dim, hidden_dim)

        # GRU dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Latent predictor: hidden → latent_{t+1}
        self.latent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Reward predictor: latent → scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

        # Online learning state
        self._buffer: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=buffer_size
        )
        self._optimizer = torch.optim.Adam(self.parameters(), lr=online_lr)
        self._online_batch = online_batch

        # GRU hidden (for rollout continuity — separate from LatentCore GRU)
        self._hidden: Optional[torch.Tensor] = None

    # ── Forward (dynamics rollout) ────────────────────────────────────────

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next latent, reward, and new hidden.

        Parameters
        ----------
        latent : (latent_dim,) or (B, latent_dim)
        action : (action_dim,) or (B, action_dim)
        hidden : (hidden_dim,) or (B, hidden_dim) — None → zeros

        Returns
        -------
        next_latent : same shape as latent
        reward_pred : scalar or (B,)
        new_hidden  : same shape as hidden input
        """
        batched = latent.dim() == 2
        if not batched:
            latent = latent.unsqueeze(0)
            action = action.unsqueeze(0)

        if hidden is None:
            hidden = torch.zeros(latent.shape[0], self.hidden_dim, device=latent.device)
        elif hidden.dim() == 1:
            hidden = hidden.unsqueeze(0).expand(latent.shape[0], -1)

        x = torch.cat([latent, action], dim=-1)     # (B, latent+action)
        x = self.input_proj(x)                       # (B, hidden_dim)
        new_hidden = self.gru(x, hidden)             # (B, hidden_dim)

        next_latent = self.latent_head(new_hidden)   # (B, latent_dim)
        reward_pred = self.reward_head(latent).squeeze(-1)  # (B,)

        if not batched:
            return next_latent.squeeze(0), reward_pred.squeeze(0), new_hidden.squeeze(0)
        return next_latent, reward_pred, new_hidden

    # ── Online update ─────────────────────────────────────────────────────

    def observe(
        self,
        latent_t: torch.Tensor,
        action_t: torch.Tensor,
        latent_next: torch.Tensor,
        reward_t: torch.Tensor,
    ) -> float:
        """Store a transition and run one online learning step.

        Returns the current prediction loss (for monitoring).
        """
        self._buffer.append((
            latent_t.detach(),
            action_t.detach(),
            latent_next.detach(),
            reward_t.detach() if reward_t.dim() > 0 else reward_t.detach().unsqueeze(0),
        ))

        if len(self._buffer) < self._online_batch:
            return 0.0

        # Sample mini-batch from buffer
        indices = torch.randperm(len(self._buffer))[: self._online_batch]
        batch = [self._buffer[i] for i in indices.tolist()]
        b_lat = torch.stack([b[0] for b in batch])
        b_act = torch.stack([b[1] for b in batch])
        b_next = torch.stack([b[2] for b in batch])
        b_rew = torch.stack([b[3].squeeze() for b in batch])

        self._optimizer.zero_grad()
        pred_next, pred_rew, _ = self(b_lat, b_act)
        loss_dyn = F.mse_loss(pred_next, b_next)
        loss_rew = F.mse_loss(pred_rew, b_rew)
        loss = loss_dyn + 0.5 * loss_rew
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self._optimizer.step()
        return float(loss.item())

    # ── Rollout (for MPPI planner) ────────────────────────────────────────

    def rollout(
        self,
        latent: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out world model for H steps.

        Parameters
        ----------
        latent     : (B, latent_dim) — batched starting latent states
        action_seq : (B, H, action_dim)

        Returns
        -------
        cumulative_reward : (B,)
        final_latent      : (B, latent_dim)
        """
        B, H, _ = action_seq.shape
        hidden = torch.zeros(B, self.hidden_dim, device=latent.device)
        cum_reward = torch.zeros(B, device=latent.device)
        gamma = 0.95

        lat = latent
        for t in range(H):
            lat, r, hidden = self(lat, action_seq[:, t], hidden)
            cum_reward = cum_reward + (gamma ** t) * r

        return cum_reward, lat

    # ── Episode management ────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Clear buffer and GRU hidden state at episode start."""
        self._buffer.clear()
        self._hidden = None

    # ── Weight init ───────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.kaiming_uniform_(p, nonlinearity="linear")
            elif "bias" in name:
                nn.init.zeros_(p)
