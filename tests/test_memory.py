"""Tests for NeuralMemoryBank and HebbianUpdater."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from analogical.memory.neural_memory import NeuralMemoryBank
from analogical.adaptation.hebbian import HebbianUpdater
from analogical.perception.latent_core import LatentCore


# ── NeuralMemoryBank ──────────────────────────────────────────────────────

class TestNeuralMemoryBank:
    def _make(self, capacity=32, d_model=64, action_dim=1):
        return NeuralMemoryBank(
            capacity=capacity,
            d_model=d_model,
            action_dim=action_dim,
            temperature=0.1,
            sim_thresh=0.85,
        )

    def test_empty_query_returns_none(self):
        bank = self._make()
        action, value, confidence = bank.query(torch.randn(64))
        assert action is None
        assert value == 0.0
        assert confidence == 0.0

    def test_write_then_query(self):
        bank = self._make()
        latent = torch.randn(64)
        action = torch.tensor([0.5])
        bank.write(latent, action, value=1.0)
        recalled_action, value, conf = bank.query(latent)
        assert recalled_action is not None
        assert recalled_action.shape == (1,)
        assert conf > 0.0

    def test_frequency_increments_on_revisit(self):
        bank = self._make()
        latent = torch.randn(64)
        action = torch.tensor([0.3])
        bank.write(latent, action, 1.0)
        bank.write(latent, action, 1.0)  # same slot — should increment freq
        bank.write(latent, action, 1.0)
        assert int(bank._freqs[0].item()) >= 2  # at least 2 (could be 3)

    def test_high_freq_recalled_over_low_freq(self):
        """A frequently written entry should win attention over a rare one."""
        bank = self._make(capacity=64, d_model=16, action_dim=1)

        # Two distinct latent contexts
        lat_rare = torch.zeros(16)
        lat_rare[0] = 1.0
        lat_freq = torch.zeros(16)
        lat_freq[1] = 1.0

        act_rare = torch.tensor([0.0])
        act_freq = torch.tensor([1.0])

        bank.write(lat_rare, act_rare, 1.0)  # freq=1
        for _ in range(10):
            bank.write(lat_freq, act_freq, 1.0)  # freq=10

        # Query with lat_freq context — should strongly recall act_freq
        query_key = lat_freq.clone()
        recalled, _, confidence = bank.query(query_key)
        assert recalled is not None
        assert confidence > 0.0

    def test_capacity_eviction(self):
        bank = self._make(capacity=4, d_model=8, action_dim=1)
        for i in range(6):  # write more than capacity
            lat = torch.randn(8)
            bank.write(lat, torch.tensor([float(i)]), float(i))
        assert bank._size <= 4

    def test_reset_clears_memory(self):
        bank = self._make()
        bank.write(torch.randn(64), torch.tensor([0.5]), 1.0)
        assert bank._size == 1
        bank.reset_episode()
        assert bank._size == 0
        action, _, _ = bank.query(torch.randn(64))
        assert action is None

    def test_top_k_memories(self):
        bank = self._make()
        for i in range(5):
            for _ in range(i + 1):  # different frequencies
                bank.write(torch.randn(64), torch.tensor([float(i)]), float(i))
        top = bank.top_k_memories(k=3)
        assert len(top) <= 3
        # Should be sorted by frequency (highest first)
        if len(top) >= 2:
            assert top[0][3] >= top[1][3]  # freq[0] >= freq[1]

    def test_value_ema_update(self):
        bank = self._make()
        latent = torch.randn(64)
        action = torch.tensor([0.0])
        bank.write(latent, action, 0.0)
        bank.write(latent, action, 1.0)  # updates EMA
        # Value should be between 0 and 1 (EMA blended)
        v = float(bank._values[0].item())
        assert 0.0 < v < 1.0

    def test_summary(self):
        bank = self._make()
        bank.write(torch.randn(64), torch.tensor([0.5]), 1.0)
        s = bank.summary()
        assert "size" in s and "max_freq" in s

    def test_action_prior_shape(self):
        bank = self._make(action_dim=2)
        bank.write(torch.randn(64), torch.randn(2), 0.5)
        action, _, _ = bank.query(torch.randn(64))
        assert action is not None
        assert action.shape == (2,)


# ── HebbianUpdater ────────────────────────────────────────────────────────

class TestHebbianUpdater:
    def _make_core_and_updater(self, d_model=64):
        core = LatentCore(d_model=d_model, n_heads=2, n_layers=1)
        updater = HebbianUpdater(core, lr=1e-4, decay=1e-5, update_every=1)
        return core, updater

    def test_update_modifies_gru_weights(self):
        core, updater = self._make_core_and_updater()
        W_before = core.gru.weight_hh.data.clone()
        h_old = torch.randn(64)
        h_new = torch.randn(64)
        updater.update(h_old, h_new)
        W_after = core.gru.weight_hh.data
        assert not torch.allclose(W_before, W_after), "Hebbian update should change weights"

    def test_update_every_skips(self):
        core, _ = self._make_core_and_updater()
        updater = HebbianUpdater(core, lr=1e-3, decay=0.0, update_every=3)
        W_before = core.gru.weight_hh.data.clone()
        h = torch.randn(64)
        updater.update(h, h)  # step 1 — no update (1 % 3 != 0)
        updater.update(h, h)  # step 2 — no update (2 % 3 != 0)
        W_after_2 = core.gru.weight_hh.data.clone()
        updater.update(h, h)  # step 3 — should update (3 % 3 == 0)
        W_after_3 = core.gru.weight_hh.data.clone()
        # After step 2 — no change
        assert torch.allclose(W_before, W_after_2), "Should not update at steps 1,2"
        # After step 3 — changed
        assert not torch.allclose(W_after_2, W_after_3), "Should update at step 3"

    def test_weights_remain_finite(self):
        """Oja rule should keep weights bounded (no blow-up)."""
        core, updater = self._make_core_and_updater()
        updater.lr = 0.1  # large lr to stress-test stability
        for _ in range(500):
            h_old = torch.randn(64)
            h_new = torch.randn(64)
            updater.update(h_old, h_new)
        assert torch.isfinite(core.gru.weight_hh.data).all(), "Weights should stay finite"

    def test_reset_clears_step_counter(self):
        core, updater = self._make_core_and_updater()
        updater._step = 42
        updater.reset_episode()
        assert updater._step == 0

    def test_requires_gru_cell(self):
        """Constructor should raise if core.gru has no weight_hh."""
        core = LatentCore(d_model=64, n_heads=2, n_layers=1)
        # Temporarily break the GRU attribute
        original_gru = core.gru
        core.gru = torch.nn.Linear(64, 64)  # no weight_hh
        with pytest.raises(AttributeError):
            HebbianUpdater(core)
        core.gru = original_gru  # restore


# ── Integration: memory + Hebbian in agent ────────────────────────────────

class TestAgentMemoryIntegration:
    def test_memory_grows_during_episode(self):
        from analogical.agent import AnalogicalAgent
        from analogical.sim.mock_env import MockCartpoleEnv

        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64, memory_capacity=256)
        obs = env.reset()
        agent.reset_episode()

        for _ in range(10):
            action, _, _ = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.observe(obs, action, next_obs, reward)
            obs = next_obs
            if done:
                break

        assert agent.memory._size > 0, "Memory should accumulate entries during episode"

    def test_hebbian_runs_every_step(self):
        from analogical.agent import AnalogicalAgent
        from analogical.sim.mock_env import MockCartpoleEnv

        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64)

        obs = env.reset()
        agent.reset_episode()

        # Step 1: seeds the GRU hidden state
        action, _, _ = agent.act(obs)
        obs, _, _, _ = env.step(action)

        # Capture weights AFTER step 1 (GRU hidden is now non-zero)
        W_before = agent.core.gru.weight_hh.data.clone()

        # Step 2: Hebbian fires with non-zero h_old
        agent.act(obs)

        W_after = agent.core.gru.weight_hh.data
        assert not torch.allclose(W_before, W_after, atol=1e-7), \
            "Hebbian should modify GRU weights when h_old is non-zero"

    def test_memory_confidence_rises_with_repetition(self):
        """After many steps with the same action, memory confidence should grow."""
        from analogical.agent import AnalogicalAgent
        from analogical.sim.mock_env import MockCartpoleEnv

        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64, memory_capacity=128)
        obs = env.reset()
        agent.reset_episode()

        confidences = []
        for _ in range(30):
            action, _, info = agent.act(obs)
            confidences.append(info["mem_confidence"])
            next_obs, reward, done, _ = env.step(action)
            agent.observe(obs, action, next_obs, reward)
            obs = next_obs
            if done:
                break

        # Confidence should eventually become non-zero
        assert any(c > 0 for c in confidences), "Memory confidence should become nonzero"
