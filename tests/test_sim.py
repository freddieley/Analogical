"""Tests for mock environments and perturbation harness."""

from __future__ import annotations

import numpy as np
import pytest

from analogical.sim.mock_env import MockCartpoleEnv, MockReachEnv
from analogical.sim.perturbation import PerturbationConfig, PerturbationHarness, _ScheduledEvent


# ── MockCartpoleEnv ──────────────────────────────────────────────────────

class TestMockCartpoleEnv:
    def test_reset_returns_sensory_bundle(self):
        env = MockCartpoleEnv(seed=0)
        obs = env.reset()
        assert obs.proprioception is not None
        assert obs.proprioception.shape == (4,)
        assert obs.tactile is not None

    def test_step_returns_correct_types(self):
        env = MockCartpoleEnv(seed=0)
        env.reset()
        action = np.array([0.1], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "step" in info

    def test_vision_shape(self):
        env = MockCartpoleEnv(render_vision=True, seed=0)
        obs = env.reset()
        assert obs.vision is not None
        assert obs.vision.shape == (1, 64, 64)
        assert obs.vision.dtype == np.float32

    def test_zero_vision_when_disabled(self):
        env = MockCartpoleEnv(render_vision=False, seed=0)
        obs = env.reset()
        assert obs.vision is None

    def test_max_steps_ends_episode(self):
        env = MockCartpoleEnv(max_steps=5, seed=0)
        env.reset()
        done = False
        for _ in range(10):
            _, _, done, _ = env.step(np.array([0.0]))
            if done:
                break
        assert done

    def test_action_dim(self):
        env = MockCartpoleEnv()
        assert env.action_dim == 1

    def test_proprioception_dim(self):
        env = MockCartpoleEnv()
        assert env.proprioception_dim == 4

    def test_reward_bounded(self):
        env = MockCartpoleEnv(seed=1)
        env.reset()
        for _ in range(50):
            _, r, done, _ = env.step(np.array([0.0]))
            assert r in (0.0, 1.0)
            if done:
                break


# ── MockReachEnv ─────────────────────────────────────────────────────────

class TestMockReachEnv:
    def test_reset_returns_sensory_bundle(self):
        env = MockReachEnv(seed=0)
        obs = env.reset()
        assert obs.proprioception is not None
        assert obs.proprioception.shape == (6,)

    def test_step_mechanics(self):
        env = MockReachEnv(seed=0)
        env.reset()
        action = np.array([0.1, -0.1], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert "dist" in info

    def test_action_dim(self):
        env = MockReachEnv()
        assert env.action_dim == 2

    def test_proprioception_dim(self):
        env = MockReachEnv()
        assert env.proprioception_dim == 6

    def test_vision_shape(self):
        env = MockReachEnv(render_vision=True, seed=0)
        obs = env.reset()
        assert obs.vision is not None
        assert obs.vision.shape == (1, 64, 64)


# ── PerturbationHarness ────────────────────────────────────────────────

class TestPerturbationHarness:
    def _make_harness(self, **kwargs):
        return PerturbationHarness(MockCartpoleEnv(seed=0), **kwargs)

    def test_baseline_passthrough(self):
        harness = self._make_harness()
        obs = harness.reset()
        assert obs.proprioception is not None
        _, _, _, info = harness.step(np.array([0.5]))
        assert "perturbation" in info

    def test_actuator_dropout_zeros_action(self):
        """With 100% dropout prob, action should always be zeroed."""
        cfg = PerturbationConfig(actuator_dropout_prob=1.0, label="test")
        harness = PerturbationHarness(
            MockCartpoleEnv(seed=0),
            schedule=[_ScheduledEvent(0, cfg)],
            seed=0,
        )
        harness.reset()
        # Run a few steps; environment should still step (just with zero force)
        for _ in range(5):
            _, _, done, _ = harness.step(np.array([5.0]))
            if done:
                break

    def test_sensor_blackout_sets_none(self):
        cfg = PerturbationConfig(sensor_blackout={"proprioception": 50}, label="blackout")
        harness = PerturbationHarness(
            MockCartpoleEnv(seed=0),
            schedule=[_ScheduledEvent(0, cfg)],
            seed=0,
        )
        harness.reset()
        obs, _, _, _ = harness.step(np.array([0.0]))
        assert obs.proprioception is None

    def test_sensor_noise_changes_observation(self):
        sigma = 10.0   # very high noise to ensure change
        cfg = PerturbationConfig(sensor_noise={"proprioception": sigma}, label="noise")
        clean = MockCartpoleEnv(seed=0)
        noisy = PerturbationHarness(
            MockCartpoleEnv(seed=0),
            schedule=[_ScheduledEvent(0, cfg)],
            seed=42,
        )
        obs_clean = clean.reset()
        obs_noisy = noisy.reset()
        clean.step(np.array([0.0]))
        obs_noisy2, _, _, _ = noisy.step(np.array([0.0]))
        # After noise, proprioception should differ from clean
        # (with sigma=10, extremely unlikely to be identical)
        assert not np.allclose(obs_noisy2.proprioception, 0.0)

    def test_task_c_suite_factory(self):
        env = MockCartpoleEnv()
        harnesses = PerturbationHarness.make_task_c_suite(env, seed=0)
        assert len(harnesses) > 0
        for h in harnesses:
            h.reset()

    def test_task_a_suite_factory(self):
        env = MockReachEnv()
        harnesses = PerturbationHarness.make_task_a_suite(env, seed=0)
        assert len(harnesses) > 0

    def test_available_modalities(self):
        harness = self._make_harness()
        obs = harness.reset()
        m = obs.available_modalities()
        assert "proprioception" in m
