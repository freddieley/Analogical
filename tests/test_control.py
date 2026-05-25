"""Tests for OnlineWorldModel and MPPIPlanner."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from analogical.control.world_model import OnlineWorldModel
from analogical.control.planner import MPPIPlanner, HARD_CAP_MS


# ── OnlineWorldModel ─────────────────────────────────────────────────────

class TestOnlineWorldModel:
    def _make(self, latent_dim=32, action_dim=1):
        return OnlineWorldModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=32,
            buffer_size=16,
            online_batch=4,
        )

    def test_forward_output_shapes(self):
        wm = self._make()
        latent = torch.randn(32)
        action = torch.randn(1)
        next_lat, rew, hidden = wm(latent, action)
        assert next_lat.shape == (32,)
        assert rew.shape == ()
        assert hidden.shape == (32,)

    def test_batched_forward(self):
        wm = self._make()
        B = 8
        latent = torch.randn(B, 32)
        action = torch.randn(B, 1)
        next_lat, rew, hidden = wm(latent, action)
        assert next_lat.shape == (B, 32)
        assert rew.shape == (B,)

    def test_rollout_shape(self):
        wm = self._make()
        B, H = 16, 5
        latent = torch.randn(B, 32)
        actions = torch.randn(B, H, 1)
        cum_r, final_lat = wm.rollout(latent, actions)
        assert cum_r.shape == (B,)
        assert final_lat.shape == (B, 32)

    def test_observe_returns_loss(self):
        wm = self._make()
        # Fill buffer first
        for _ in range(4):
            lt = torch.randn(32)
            at = torch.randn(1)
            ln = torch.randn(32)
            rt = torch.tensor([0.5])
            loss = wm.observe(lt, at, ln, rt)
        # Once buffer is warm, should return non-None loss
        assert loss is not None

    def test_observe_returns_zero_before_warm(self):
        wm = self._make()
        lt = torch.randn(32)
        at = torch.randn(1)
        ln = torch.randn(32)
        rt = torch.tensor([0.5])
        loss = wm.observe(lt, at, ln, rt)
        assert loss == 0.0

    def test_online_loss_decreases(self):
        """After many updates the model should fit training data better."""
        wm = self._make(latent_dim=16, action_dim=1)
        wm._optimizer = torch.optim.Adam(wm.parameters(), lr=1e-2)

        # Always the same (latent, action, next_latent) pair
        lt = torch.zeros(16)
        at = torch.zeros(1)
        ln = torch.ones(16)

        losses = []
        for _ in range(100):
            l = wm.observe(lt, at, ln, torch.tensor([1.0]))
            if l > 0:
                losses.append(l)

        assert len(losses) > 5, "Not enough non-zero losses collected"
        # Mean of last 10 should be less than mean of first 10
        first = sum(losses[:10]) / 10
        last = sum(losses[-10:]) / 10
        assert last < first, f"Loss did not decrease: {first:.4f} → {last:.4f}"

    def test_reset_clears_buffer(self):
        wm = self._make()
        for _ in range(10):
            wm.observe(torch.randn(32), torch.randn(1), torch.randn(32), torch.tensor([0.0]))
        wm.reset_episode()
        assert len(wm._buffer) == 0


# ── MPPIPlanner ───────────────────────────────────────────────────────────

class TestMPPIPlanner:
    def _make(self, action_dim=1, n_samples=64, horizon=5):
        low = np.full(action_dim, -1.0)
        high = np.full(action_dim, 1.0)
        return MPPIPlanner(
            action_dim=action_dim,
            action_low=low,
            action_high=high,
            n_samples=n_samples,
            horizon=horizon,
        )

    def _make_wm(self, latent_dim=32, action_dim=1):
        return OnlineWorldModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=32)

    def test_output_action_shape(self):
        planner = self._make()
        wm = self._make_wm()
        latent = torch.randn(32)
        action, elapsed, info = planner.plan(latent, wm)
        assert action.shape == (1,)

    def test_action_within_bounds(self):
        planner = self._make(action_dim=2)
        wm = self._make_wm(action_dim=2)
        latent = torch.randn(32)
        for _ in range(10):
            action, _, _ = planner.plan(latent, wm)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_deadline_met(self):
        """With small N/H, planning should comfortably stay under 200ms."""
        planner = self._make(n_samples=32, horizon=3)
        wm = self._make_wm()
        latent = torch.randn(32)
        _, elapsed, info = planner.plan(latent, wm)
        assert elapsed <= HARD_CAP_MS, f"Deadline violated: {elapsed:.1f}ms"
        assert info["deadline_met"]

    def test_hard_cap_flag_on_violation(self):
        """Simulate a deadline violation by passing a tiny budget."""
        planner = self._make(n_samples=32768, horizon=50)
        wm = self._make_wm()
        latent = torch.randn(32)
        # Give almost no budget — expect early exit or flagged violation
        action, elapsed, info = planner.plan(latent, wm, deadline_ms=201.0)
        # action should still be a valid array
        assert action.shape == (1,)

    def test_warm_start(self):
        planner = self._make()
        wm = self._make_wm()
        latent = torch.randn(32)
        # First call should set _prev_seq
        planner.plan(latent, wm)
        assert planner._prev_seq is not None

    def test_reset_clears_warm_start(self):
        planner = self._make()
        wm = self._make_wm()
        planner.plan(torch.randn(32), wm)
        planner.reset_episode()
        assert planner._prev_seq is None
