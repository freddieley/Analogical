"""Pure-numpy mock environments — no MuJoCo required.

Tasks
-----
MockCartpoleEnv  (Task C) — balance/control under actuator perturbation.
    State : [x, x_dot, theta, theta_dot]
    Action: scalar force on cart, clipped to [-action_max, +action_max]
    Reward: 1.0 per step while |theta| < SUCCESS_ANGLE, 0 otherwise
    Done  : |theta| >= FAIL_ANGLE or |x| >= x_limit

MockReachEnv     (Task A) — 2-D planar reaching under sensor dropout/noise.
    State : [q1, q2, dq1, dq2, target_x, target_y]   (two-joint arm + target)
    Action: [torque1, torque2]
    Reward: -distance(tip, target), with +10 bonus on reach (<= reach_tol)
    Done  : reach success or max_steps exceeded
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import BaseEnv, SensoryBundle

# ─── CartPole constants ────────────────────────────────────────────────────
_CART_MASS = 1.0
_POLE_MASS = 0.1
_POLE_HALF_LEN = 0.5
_GRAVITY = 9.81
_DT = 0.02            # 50 Hz physics
_X_LIMIT = 2.4
_SUCCESS_ANGLE = np.deg2rad(12.0)   # balanced if within this
_FAIL_ANGLE = np.deg2rad(24.0)      # episode ends here

# ─── Reach constants ───────────────────────────────────────────────────────
_LINK_LEN = [0.5, 0.4]   # arm segment lengths
_DT_REACH = 0.02
_REACH_TOL = 0.05         # m — success radius
_REACH_MAX_STEPS = 400

# ─── Rendering ────────────────────────────────────────────────────────────
_IMG_SIZE = 64


def _render_cartpole(state: np.ndarray) -> np.ndarray:
    """Return (1, 64, 64) float32 image of cartpole state."""
    img = np.ones((_IMG_SIZE, _IMG_SIZE), dtype=np.float32)
    x, _, theta, _ = state

    # Cart centre pixel (x in [-2.4, 2.4] → [4, 60])
    cx = int(np.clip((_IMG_SIZE / 2) + x * (_IMG_SIZE / 2) / _X_LIMIT, 4, _IMG_SIZE - 5))
    cy = _IMG_SIZE // 2

    # Cart rectangle (9 × 5)
    img[cy - 2 : cy + 3, max(0, cx - 4) : min(_IMG_SIZE, cx + 5)] = 0.2

    # Pole tip
    pole_px = int(_POLE_HALF_LEN * 2 * (_IMG_SIZE / 6))  # scale
    tx = int(cx + pole_px * np.sin(theta))
    ty = int(cy - pole_px * np.cos(theta))
    tx = int(np.clip(tx, 0, _IMG_SIZE - 1))
    ty = int(np.clip(ty, 0, _IMG_SIZE - 1))

    # Bresenham line
    _draw_line(img, cx, cy, tx, ty, value=0.0)

    return img[np.newaxis]  # (1, H, W)


def _render_reach(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return (1, 64, 64) float32 image of 2-joint arm + target."""
    img = np.ones((_IMG_SIZE, _IMG_SIZE), dtype=np.float32)

    def _to_px(pt: np.ndarray) -> tuple[int, int]:
        # World coords: arm centred at (0, 0), workspace ~[-1, 1]^2
        px = int(np.clip(pt[0] * 20 + _IMG_SIZE // 2, 0, _IMG_SIZE - 1))
        py = int(np.clip(-pt[1] * 20 + _IMG_SIZE // 2, 0, _IMG_SIZE - 1))
        return px, py

    # Joint positions
    origin = np.zeros(2)
    j1 = origin + _LINK_LEN[0] * np.array([np.cos(q[0]), np.sin(q[0])])
    j2 = j1 + _LINK_LEN[1] * np.array([
        np.cos(q[0] + q[1]), np.sin(q[0] + q[1])
    ])

    o_px = _to_px(origin)
    j1_px = _to_px(j1)
    j2_px = _to_px(j2)
    t_px = _to_px(target)

    _draw_line(img, o_px[0], o_px[1], j1_px[0], j1_px[1], value=0.0)
    _draw_line(img, j1_px[0], j1_px[1], j2_px[0], j2_px[1], value=0.0)

    # Target marker (3×3 block)
    ty0, ty1 = max(0, t_px[1] - 1), min(_IMG_SIZE, t_px[1] + 2)
    tx0, tx1 = max(0, t_px[0] - 1), min(_IMG_SIZE, t_px[0] + 2)
    img[ty0:ty1, tx0:tx1] = 0.5

    return img[np.newaxis]  # (1, H, W)


def _draw_line(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: float = 0.0
) -> None:
    """Bresenham line draw in-place (img is H×W)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    H, W = img.shape
    while True:
        if 0 <= y0 < H and 0 <= x0 < W:
            img[y0, x0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


# ═══════════════════════════════════════════════════════════════════════════
# Task C — Balance/control under actuator perturbation
# ═══════════════════════════════════════════════════════════════════════════

class MockCartpoleEnv(BaseEnv):
    """Continuous-action cartpole. Supports actuator perturbation injection."""

    task_name = "balance_actuator_perturbation"

    def __init__(
        self,
        max_steps: int = 500,
        render_vision: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._max_steps = max_steps
        self._render_vision = render_vision
        self._rng = np.random.default_rng(seed)
        self._state: np.ndarray = np.zeros(4, dtype=np.float32)
        self._step_count: int = 0
        self._t_ms: float = 0.0

    # ── BaseEnv interface ────────────────────────────────────────────────

    def reset(self) -> SensoryBundle:
        self._state = self._rng.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self._step_count = 0
        self._t_ms = time.monotonic() * 1000.0
        return self._observe()

    def step(
        self, action: np.ndarray, *, _override_force: Optional[float] = None
    ) -> tuple[SensoryBundle, float, bool, dict]:
        force = float(np.clip(action[0], -10.0, 10.0))
        if _override_force is not None:
            force = _override_force

        self._state = _cartpole_dynamics(self._state, force)
        self._step_count += 1
        self._t_ms += _DT * 1000.0

        x, _, theta, _ = self._state
        balanced = abs(theta) < _SUCCESS_ANGLE
        failed = abs(theta) >= _FAIL_ANGLE or abs(x) >= _X_LIMIT
        done = failed or self._step_count >= self._max_steps

        reward = 1.0 if balanced else 0.0
        return self._observe(), reward, done, {
            "step": self._step_count,
            "balanced": balanced,
            "failed": failed,
        }

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-10.0], dtype=np.float32), np.array([10.0], dtype=np.float32)

    @property
    def proprioception_dim(self) -> int:
        return 4

    # ── Internal helpers ─────────────────────────────────────────────────

    def _observe(self) -> SensoryBundle:
        x, _, theta, _ = self._state
        tactile = np.array([abs(x) / _X_LIMIT], dtype=np.float32)  # contact proxy
        vision = _render_cartpole(self._state) if self._render_vision else None
        return SensoryBundle(
            proprioception=self._state.copy(),
            vision=vision,
            tactile=tactile,
            timestamp_ms=self._t_ms,
        )


def _cartpole_dynamics(state: np.ndarray, force: float) -> np.ndarray:
    """4th-order Runge-Kutta integration of cartpole ODEs."""
    x, xd, th, thd = state

    def _deriv(s: np.ndarray, f: float) -> np.ndarray:
        _, xdot, theta, thetadot = s
        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        total_mass = _CART_MASS + _POLE_MASS
        pole_mass_len = _POLE_MASS * _POLE_HALF_LEN

        tmp = (f + pole_mass_len * thetadot ** 2 * sin_th) / total_mass
        theta_acc = (_GRAVITY * sin_th - cos_th * tmp) / (
            _POLE_HALF_LEN * (4.0 / 3.0 - _POLE_MASS * cos_th ** 2 / total_mass)
        )
        x_acc = tmp - pole_mass_len * theta_acc * cos_th / total_mass
        return np.array([xdot, x_acc, thetadot, theta_acc], dtype=np.float64)

    s = state.astype(np.float64)
    k1 = _deriv(s, force)
    k2 = _deriv(s + 0.5 * _DT * k1, force)
    k3 = _deriv(s + 0.5 * _DT * k2, force)
    k4 = _deriv(s + _DT * k3, force)
    s_new = s + (_DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s_new.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Task A — Reaching under sensor dropout / noise
# ═══════════════════════════════════════════════════════════════════════════

class MockReachEnv(BaseEnv):
    """2-joint planar arm reaching task. Supports sensor dropout/noise injection."""

    task_name = "reach_sensor_perturbation"

    def __init__(
        self,
        max_steps: int = _REACH_MAX_STEPS,
        render_vision: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._max_steps = max_steps
        self._render_vision = render_vision
        self._rng = np.random.default_rng(seed)
        # State: [q1, q2, dq1, dq2] + fixed target stored separately
        self._q = np.zeros(2, dtype=np.float32)
        self._dq = np.zeros(2, dtype=np.float32)
        self._target = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._t_ms = 0.0
        self._success = False

    # ── BaseEnv interface ────────────────────────────────────────────────

    def reset(self) -> SensoryBundle:
        self._q = self._rng.uniform(-np.pi / 4, np.pi / 4, size=2).astype(np.float32)
        self._dq = np.zeros(2, dtype=np.float32)
        # Random reachable target
        angle = self._rng.uniform(0, 2 * np.pi)
        radius = self._rng.uniform(0.3, sum(_LINK_LEN) * 0.9)
        self._target = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
        )
        self._step_count = 0
        self._t_ms = time.monotonic() * 1000.0
        self._success = False
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[SensoryBundle, float, bool, dict]:
        torques = np.clip(action, -2.0, 2.0)
        self._q, self._dq = _arm_dynamics(self._q, self._dq, torques)
        self._step_count += 1
        self._t_ms += _DT_REACH * 1000.0

        tip = _arm_tip(self._q)
        dist = float(np.linalg.norm(tip - self._target))
        reached = dist <= _REACH_TOL
        done = reached or self._step_count >= self._max_steps

        if reached and not self._success:
            self._success = True

        reward = -dist + (10.0 if reached else 0.0)
        return self._observe(), reward, done, {
            "step": self._step_count,
            "dist": dist,
            "reached": reached,
        }

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(2, -2.0, dtype=np.float32), np.full(2, 2.0, dtype=np.float32)

    @property
    def proprioception_dim(self) -> int:
        return 6  # [q1, q2, dq1, dq2, target_x, target_y]

    # ── Internal helpers ─────────────────────────────────────────────────

    def _observe(self) -> SensoryBundle:
        proprio = np.concatenate([self._q, self._dq, self._target]).astype(np.float32)
        tip = _arm_tip(self._q)
        tactile = np.array(
            [float(np.linalg.norm(tip - self._target) < _REACH_TOL)], dtype=np.float32
        )
        vision = _render_reach(self._q, self._target) if self._render_vision else None
        return SensoryBundle(
            proprioception=proprio,
            vision=vision,
            tactile=tactile,
            timestamp_ms=self._t_ms,
        )


def _arm_tip(q: np.ndarray) -> np.ndarray:
    j1 = _LINK_LEN[0] * np.array([np.cos(q[0]), np.sin(q[0])])
    j2 = j1 + _LINK_LEN[1] * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
    return j2.astype(np.float32)


def _arm_dynamics(
    q: np.ndarray, dq: np.ndarray, torques: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Simplified 2-link arm dynamics (unit inertia, viscous damping)."""
    # Inertia matrix (simplified: diagonal, dependent on link lengths)
    I1 = _LINK_LEN[0] ** 2 + _LINK_LEN[1] ** 2
    I2 = _LINK_LEN[1] ** 2
    damping = 0.1

    ddq = np.array([
        (torques[0] - damping * dq[0]) / I1,
        (torques[1] - damping * dq[1]) / I2,
    ], dtype=np.float32)

    dq_new = dq + ddq * _DT_REACH
    dq_new = np.clip(dq_new, -10.0, 10.0)
    q_new = q + dq_new * _DT_REACH
    return q_new.astype(np.float32), dq_new.astype(np.float32)
