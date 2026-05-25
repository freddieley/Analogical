"""Tests for the full AnalogicalAgent and end-to-end benchmark loop."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from analogical.agent import AnalogicalAgent
from analogical.sim.mock_env import MockCartpoleEnv, MockReachEnv
from analogical.sim.perturbation import PerturbationConfig, PerturbationHarness, _ScheduledEvent
from analogical.benchmark.tasks import BalanceTask, ReachTask, PerturbationSuite
from analogical.benchmark.runner import BenchmarkRunner
from analogical.benchmark.metrics import EpisodeMetrics, SuiteMetrics


# ── Agent construction ────────────────────────────────────────────────────

class TestAgentConstruction:
    def test_from_cartpole_env(self):
        env = MockCartpoleEnv(render_vision=False)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        assert agent.action_dim == 1
        assert agent.d_model == 64

    def test_from_reach_env(self):
        env = MockReachEnv(render_vision=False)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        assert agent.action_dim == 2

    def test_no_pretrained_weights(self):
        env = MockCartpoleEnv(render_vision=False)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        # All finite weights — not pretrained (no NaN or Inf)
        for name, p in agent.core.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite: {name}"


# ── Agent act / observe loop ──────────────────────────────────────────────

class TestAgentActObserve:
    def _make_agent_env(self):
        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        return agent, env

    def test_act_returns_valid_action(self):
        agent, env = self._make_agent_env()
        obs = env.reset()
        agent.reset_episode()
        action, elapsed_ms, info = agent.act(obs)
        assert action.shape == (1,)
        assert isinstance(elapsed_ms, float)
        assert "deadline_met" in info

    def test_act_respects_200ms_deadline(self):
        agent, env = self._make_agent_env()
        obs = env.reset()
        agent.reset_episode()
        _, elapsed_ms, info = agent.act(obs)
        assert elapsed_ms <= 200.0, f"act() exceeded 200ms: {elapsed_ms:.1f}ms"

    def test_observe_returns_loss(self):
        agent, env = self._make_agent_env()
        obs = env.reset()
        agent.reset_episode()
        action, _, _ = agent.act(obs)
        next_obs, reward, _, _ = env.step(action)
        # observe may return None if buffer not warm
        result = agent.observe(obs, action, next_obs, reward)
        assert result is None or isinstance(result, float)

    def test_multi_step_episode(self):
        agent, env = self._make_agent_env()
        obs = env.reset()
        agent.reset_episode()
        for _ in range(20):
            action, elapsed, _ = agent.act(obs)
            assert elapsed <= 200.0
            obs, _, done, _ = env.step(action)
            if done:
                break

    def test_missing_modality_does_not_crash(self):
        """Agent must handle None proprioception gracefully."""
        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        obs = env.reset()
        obs.proprioception = None   # simulate blackout
        agent.reset_episode()
        action, _, _ = agent.act(obs)
        assert action.shape == (1,)

    def test_all_modalities_missing(self):
        """All-None bundle should not crash — agent uses world model prediction."""
        from analogical.sim.base import SensoryBundle
        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        # First step with real obs to seed hidden state
        obs = env.reset()
        agent.reset_episode()
        agent.act(obs)
        # Now all-None
        empty = SensoryBundle()
        action, _, _ = agent.act(empty)
        assert action.shape == (1,)

    def test_reset_episode_clears_state(self):
        agent, env = self._make_agent_env()
        obs = env.reset()
        agent.reset_episode()
        agent.act(obs)
        agent.reset_episode()
        assert agent._last_latent is None
        assert agent._step == 0


# ── Benchmark tasks ───────────────────────────────────────────────────────

class TestBenchmarkTasks:
    def test_balance_task_scenarios(self):
        specs = BalanceTask.scenarios(seed=0)
        assert len(specs) > 0
        for spec in specs:
            assert spec.task_id == "C"

    def test_reach_task_scenarios(self):
        specs = ReachTask.scenarios(seed=0)
        assert len(specs) > 0
        for spec in specs:
            assert spec.task_id == "A"

    def test_perturbation_suite_all(self):
        specs = PerturbationSuite.all_scenarios(seed=0)
        task_ids = {s.task_id for s in specs}
        assert "A" in task_ids
        assert "C" in task_ids


# ── BenchmarkRunner ───────────────────────────────────────────────────────

class TestBenchmarkRunner:
    def _make_runner(self, episodes=1):
        return BenchmarkRunner(
            agent=None,  # will be set per-test
            episodes_per_scenario=episodes,
            verbose=False,
        )

    def test_single_scenario_run(self):
        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        specs = BalanceTask.scenarios(seed=0)[:2]   # first 2 only

        runner = BenchmarkRunner(agent=agent, episodes_per_scenario=1, verbose=False)
        suite = runner.run(specs)
        assert len(suite.episodes) == 2

    def test_suite_metrics_computed(self):
        env = MockReachEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        specs = ReachTask.scenarios(seed=0)[:2]

        runner = BenchmarkRunner(agent=agent, episodes_per_scenario=1, verbose=False)
        suite = runner.run(specs)
        assert 0.0 <= suite.success_rate <= 1.0
        assert 0.0 <= suite.timing_compliance_rate <= 1.0
        assert 0.0 <= suite.degradation_auc <= 1.0

    def test_timing_compliance_enforced(self):
        env = MockCartpoleEnv(render_vision=False, seed=0)
        agent = AnalogicalAgent.from_env(env, d_model=64)
        specs = BalanceTask.scenarios(seed=0)[:1]

        runner = BenchmarkRunner(agent=agent, episodes_per_scenario=1, verbose=False)
        suite = runner.run(specs)
        # In a fast (no-vision, small model) run, timing should be met
        for ep in suite.episodes:
            # Allow up to 10% violations in a CI-constrained environment
            assert ep.timing_compliance_rate >= 0.9, (
                f"Too many deadline violations: {ep.timing_compliance_rate:.2%}"
            )


# ── EpisodeMetrics ─────────────────────────────────────────────────────────

class TestEpisodeMetrics:
    def test_timing_compliance_all_ok(self):
        m = EpisodeMetrics("test", "C", "baseline")
        m.step_times_ms = [50.0, 100.0, 150.0]
        assert m.timing_compliance_rate == 1.0

    def test_timing_compliance_some_violations(self):
        m = EpisodeMetrics("test", "C", "test")
        m.step_times_ms = [50.0, 210.0, 100.0, 205.0]
        assert m.timing_compliance_rate == 0.5

    def test_recovery_latency_computation(self):
        m = EpisodeMetrics("test", "C", "test")
        m.perturbation_onset_step = 50
        m.recovery_step = 60
        assert m.recovery_latency_ms == pytest.approx(200.0)

    def test_recovery_latency_none_if_no_perturbation(self):
        m = EpisodeMetrics("test", "C", "baseline")
        assert m.recovery_latency_ms is None

    def test_as_dict_keys(self):
        m = EpisodeMetrics("test", "C", "baseline")
        d = m.as_dict()
        assert "task" in d and "success" in d and "timing_compliance" in d
