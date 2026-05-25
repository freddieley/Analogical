#!/usr/bin/env python3
"""run_benchmark.py — entry point for the Analogical benchmark suite.

Usage
-----
    # Both tasks (A + C) — default
    python run_benchmark.py

    # Task A only (reaching under sensor dropout/noise)
    python run_benchmark.py --task A

    # Task C only (balance under actuator perturbation)
    python run_benchmark.py --task C

    # Use MuJoCo if installed
    python run_benchmark.py --mujoco

    # More episodes per scenario (higher variance estimate)
    python run_benchmark.py --episodes 5

    # Disable vision rendering (faster on CPU)
    python run_benchmark.py --no-vision

    # Export per-episode CSV
    python run_benchmark.py --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analogical benchmark runner")
    p.add_argument("--task", choices=["A", "C", "both"], default="both",
                   help="Which task(s) to benchmark (default: both)")
    p.add_argument("--mujoco", action="store_true",
                   help="Use MuJoCo environments (requires mujoco package)")
    p.add_argument("--episodes", type=int, default=3,
                   help="Episodes per perturbation scenario (default: 3)")
    p.add_argument("--no-vision", action="store_true",
                   help="Disable visual rendering (faster; omits vision modality)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                   help="Compute device (default: auto)")
    p.add_argument("--csv", metavar="FILE",
                   help="Export per-episode metrics to a CSV file")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--quiet", action="store_true", help="Suppress per-episode output")
    return p.parse_args()


def _select_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)

    print(f"Analogical Benchmark")
    print(f"  device  : {device}")
    print(f"  task(s) : {args.task}")
    print(f"  episodes: {args.episodes} per scenario")
    print(f"  vision  : {'off' if args.no_vision else 'on'}")
    print()

    # ── Build scenario list ────────────────────────────────────────────────
    from analogical.benchmark.tasks import PerturbationSuite

    if args.mujoco:
        try:
            import mujoco  # noqa: F401
        except ImportError:
            print("ERROR: --mujoco requested but mujoco package is not installed.")
            print("       Install with: pip install mujoco")
            sys.exit(1)
        print("MuJoCo backend selected (not yet wired to benchmark — using mock).")
        print("Falling back to mock environments.\n")

    render_vision = not args.no_vision

    if args.task == "A":
        scenarios = PerturbationSuite.task_a_only(seed=args.seed)
    elif args.task == "C":
        scenarios = PerturbationSuite.task_c_only(seed=args.seed)
    else:
        scenarios = PerturbationSuite.all_scenarios(seed=args.seed)

    # Disable vision rendering in environments if requested
    if not render_vision:
        for spec in scenarios:
            inner = spec.env
            # Drill through PerturbationHarness wrappers
            while hasattr(inner, "_env"):
                inner = inner._env
            if hasattr(inner, "_render_vision"):
                inner._render_vision = False

    # ── Build agent ────────────────────────────────────────────────────────
    # Use first scenario's env to get dims
    first_env = scenarios[0].env
    inner = first_env
    while hasattr(inner, "_env"):
        inner = inner._env

    from analogical.agent import AnalogicalAgent

    agent = AnalogicalAgent.from_env(inner, d_model=256, device=device)

    print(f"Agent parameters : {sum(p.numel() for p in agent.core.parameters()):,} (core)")
    print(f"Budget level     : {agent.governor.current_level.name}")
    print()

    # ── Run benchmark ──────────────────────────────────────────────────────
    from analogical.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(
        agent=agent,
        episodes_per_scenario=args.episodes,
        verbose=not args.quiet,
    )

    t_start = time.monotonic()
    suite = runner.run(scenarios)
    elapsed = time.monotonic() - t_start

    print(f"\nTotal benchmark time: {elapsed:.1f}s")

    # ── CSV export ─────────────────────────────────────────────────────────
    if args.csv:
        rows = suite.per_episode_table()
        if rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Results written to: {args.csv}")


if __name__ == "__main__":
    main()
