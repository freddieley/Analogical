"""Benchmark runner — orchestrates agent-environment interaction.

Each scenario gets a fresh agent built from an ``agent_factory`` callable,
ensuring agents are sized correctly for their environment and start from
zero (no cross-scenario knowledge leakage).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from analogical.benchmark.metrics import EpisodeMetrics, SuiteMetrics
from analogical.benchmark.tasks import TaskSpec
from analogical.sim.perturbation import PerturbationHarness


class BenchmarkRunner:
    """Runs an agent (or fresh agents from a factory) against a list of
    TaskSpecs and collects SuiteMetrics.

    Parameters
    ----------
    agent           : a pre-built agent (used for all scenarios).
                      Ignored if ``agent_factory`` is provided.
    agent_factory   : callable(TaskSpec) -> agent.  Called once per scenario
                      so each scenario starts from a freshly-initialised agent
                      with the correct environment dimensions.  Recommended
                      when the suite mixes tasks with different state dims.
    episodes_per_scenario : int
    verbose         : print per-episode results to stdout
    """

    def __init__(
        self,
        agent=None,
        agent_factory: Optional[Callable] = None,
        episodes_per_scenario: int = 3,
        verbose: bool = True,
    ) -> None:
        if agent is None and agent_factory is None:
            raise ValueError("Provide either 'agent' or 'agent_factory'.")
        self._agent = agent
        self._factory = agent_factory
        self._eps = episodes_per_scenario
        self._verbose = verbose

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self, scenarios: list[TaskSpec]) -> SuiteMetrics:
        """Run all scenarios and return aggregated SuiteMetrics."""
        suite = SuiteMetrics()

        for spec in scenarios:
            agent = self._factory(spec) if self._factory else self._agent
            for ep_idx in range(self._eps):
                metrics = self._run_episode(spec, agent, ep_idx)
                suite.episodes.append(metrics)
                if self._verbose:
                    self._print_episode(metrics, ep_idx)

        if self._verbose:
            self._print_summary(suite)

        return suite

    # ── Episode loop ──────────────────────────────────────────────────────

    def _run_episode(self, spec: TaskSpec, agent, ep_idx: int) -> EpisodeMetrics:
        env = spec.env

        agent.reset_episode()
        obs = env.reset()

        metrics = EpisodeMetrics(
            task_name=spec.name,
            task_id=spec.task_id,
            perturbation_label=spec.perturbation_label,
        )

        # Detect perturbation onset step
        onset_step: Optional[int] = None
        if isinstance(env, PerturbationHarness):
            events = sorted(env._schedule, key=lambda e: e.onset_step)
            for event in events:
                if event.onset_step > 0 and event.config.label not in ("", "clean"):
                    onset_step = event.onset_step
                    break
        metrics.perturbation_onset_step = onset_step

        last_stable_step: Optional[int] = None

        for step in range(spec.max_steps):
            t_step_start = time.monotonic()

            action, elapsed_ms, agent_info = agent.act(obs)
            next_obs, reward, done, info = env.step(action)

            wm_loss = agent.observe(obs, action, next_obs, reward)
            if wm_loss is not None:
                metrics.wm_losses.append(wm_loss)

            total_elapsed_ms = (time.monotonic() - t_step_start) * 1000.0

            metrics.step_times_ms.append(total_elapsed_ms)
            if total_elapsed_ms > 200.0:
                metrics.timing_violations.append(total_elapsed_ms)
            metrics.total_reward += reward
            metrics.total_steps += 1

            if onset_step is not None and step >= onset_step:
                if spec.success_fn(info):
                    if last_stable_step is None:
                        last_stable_step = step
                else:
                    last_stable_step = None

            obs = next_obs

            if done:
                metrics.task_success = spec.success_fn(info)
                break
        else:
            metrics.task_success = spec.success_fn({"reached": False, "failed": False})

        if onset_step is not None and last_stable_step is not None:
            metrics.recovery_step = last_stable_step

        return metrics

    # ── Output helpers ────────────────────────────────────────────────────

    @staticmethod
    def _print_episode(m: EpisodeMetrics, ep_idx: int) -> None:
        status = "✓" if m.task_success else "✗"
        rec = (
            f"  recovery={m.recovery_latency_ms:.0f}ms"
            if m.recovery_latency_ms is not None
            else ""
        )
        print(
            f"  [{m.task_id}] {m.task_name:45s} ep{ep_idx} {status}"
            f"  timing={m.timing_compliance_rate:.2%}"
            f"  mean={m.mean_step_ms:.1f}ms"
            f"  max={m.max_step_ms:.1f}ms"
            f"{rec}"
        )

    @staticmethod
    def _print_summary(suite: SuiteMetrics) -> None:
        s = suite.summary()
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"  Episodes          : {s['n_episodes']}")
        print(f"  Success rate      : {s['success_rate']:.2%}")
        print(f"  Timing compliance : {s['timing_compliance_rate']:.2%}")
        rec = s["mean_recovery_latency_ms"]
        print(f"  Recovery latency  : {f'{rec:.0f}ms' if rec is not None else 'n/a'}")
        print(f"  Degradation AUC   : {s['degradation_auc']:.4f}")
        for tid, sr in sorted(s["per_task_success_rate"].items()):
            print(f"  Task {tid} success   : {sr:.2%}")
        print("=" * 70)

