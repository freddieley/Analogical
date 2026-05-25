"""MuJoCo environment wrapper (optional — falls back to mock if unavailable)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import BaseEnv, SensoryBundle

try:
    import mujoco  # type: ignore

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False


# ── Cartpole XML ──────────────────────────────────────────────────────────

_CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.02" integrator="RK4"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2.4 2.4"/>
      <geom type="box" size="0.2 0.1 0.05" rgba="0.8 0.4 0.1 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" rgba="0.2 0.6 0.8 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" name="slide_motor" gear="20" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# ── Reach arm XML ─────────────────────────────────────────────────────────

_REACH_XML = """
<mujoco model="reach">
  <option timestep="0.02" integrator="RK4"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03" rgba="0.4 0.4 0.8 1"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.025" rgba="0.4 0.8 0.4 1"/>
        <site name="tip" pos="0.4 0 0" size="0.02"/>
      </body>
    </body>
    <site name="target" pos="0.6 0.3 0" size="0.04" rgba="1 0 0 0.5"/>
  </worldbody>
  <actuator>
    <motor joint="j1" name="m1" gear="5" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="j2" name="m2" gear="5" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _require_mujoco() -> None:
    if not _MUJOCO_AVAILABLE:
        raise ImportError(
            "mujoco package not installed. Install with: pip install mujoco\n"
            "Or use the mock environments (MockCartpoleEnv / MockReachEnv) instead."
        )


class MuJoCoCartpoleEnv(BaseEnv):
    """MuJoCo-backed continuous cartpole (Task C).

    Falls back gracefully with an ImportError if mujoco is not installed.
    """

    task_name = "balance_actuator_perturbation_mujoco"

    def __init__(
        self,
        max_steps: int = 500,
        camera_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        _require_mujoco()
        self._max_steps = max_steps
        self._camera_name = camera_name
        self._rng = np.random.default_rng(seed)

        self._model = mujoco.MjModel.from_xml_string(_CARTPOLE_XML)
        self._data = mujoco.MjData(self._model)
        self._renderer: Optional[mujoco.Renderer] = None
        self._step_count = 0
        self._t_ms = 0.0

    def reset(self) -> SensoryBundle:
        mujoco.mj_resetData(self._model, self._data)
        init = self._rng.uniform(-0.05, 0.05, size=self._model.nq + self._model.nv)
        self._data.qpos[:] = init[: self._model.nq]
        self._data.qvel[:] = init[self._model.nq :]
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        self._t_ms = time.monotonic() * 1000.0
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[SensoryBundle, float, bool, dict]:
        self._data.ctrl[0] = float(np.clip(action[0], -1.0, 1.0))
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        self._t_ms += self._model.opt.timestep * 1000.0

        theta = float(self._data.qpos[1])
        x = float(self._data.qpos[0])
        balanced = abs(theta) < np.deg2rad(12.0)
        failed = abs(theta) >= np.deg2rad(24.0) or abs(x) >= 2.4
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
        return np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    @property
    def proprioception_dim(self) -> int:
        return self._model.nq + self._model.nv

    def _observe(self) -> SensoryBundle:
        proprio = np.concatenate([self._data.qpos, self._data.qvel]).astype(np.float32)
        tactile = np.array([abs(float(self._data.qpos[0])) / 2.4], dtype=np.float32)
        vision = self._render_frame()
        return SensoryBundle(
            proprioception=proprio,
            vision=vision,
            tactile=tactile,
            timestamp_ms=self._t_ms,
        )

    def _render_frame(self) -> Optional[np.ndarray]:
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, height=64, width=64)
            self._renderer.update_scene(self._data)
            rgb = self._renderer.render()
            gray = np.mean(rgb, axis=2, keepdims=False).astype(np.float32) / 255.0
            return gray[np.newaxis]  # (1, 64, 64)
        except Exception:
            return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()


class MuJoCoReachEnv(BaseEnv):
    """MuJoCo-backed 2-joint planar reach task (Task A)."""

    task_name = "reach_sensor_perturbation_mujoco"

    def __init__(
        self,
        max_steps: int = 400,
        seed: Optional[int] = None,
    ) -> None:
        _require_mujoco()
        self._max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        self._model = mujoco.MjModel.from_xml_string(_REACH_XML)
        self._data = mujoco.MjData(self._model)
        self._renderer: Optional[mujoco.Renderer] = None
        self._target_pos = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._t_ms = 0.0

    def reset(self) -> SensoryBundle:
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = self._rng.uniform(-0.5, 0.5, size=self._model.nq)
        self._data.qvel[:] = 0.0
        # Randomise target site position
        angle = self._rng.uniform(0, 2 * np.pi)
        r = self._rng.uniform(0.3, 0.85)
        self._target_pos = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
        tip_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self._model.site_pos[tip_id][:2] = self._target_pos
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        self._t_ms = time.monotonic() * 1000.0
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[SensoryBundle, float, bool, dict]:
        self._data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1
        self._t_ms += self._model.opt.timestep * 1000.0

        tip_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        tip_pos = self._data.site_xpos[tip_id][:2]
        dist = float(np.linalg.norm(tip_pos - self._target_pos))
        reached = dist <= 0.05
        done = reached or self._step_count >= self._max_steps
        reward = -dist + (10.0 if reached else 0.0)
        return self._observe(), reward, done, {
            "step": self._step_count,
            "dist": dist,
            "reached": reached,
        }

    @property
    def action_dim(self) -> int:
        return self._model.nu

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(self.action_dim, -1.0, dtype=np.float32), np.full(self.action_dim, 1.0, dtype=np.float32)

    @property
    def proprioception_dim(self) -> int:
        return self._model.nq + self._model.nv + 2  # joints + velocities + target

    def _observe(self) -> SensoryBundle:
        proprio = np.concatenate([
            self._data.qpos,
            self._data.qvel,
            self._target_pos,
        ]).astype(np.float32)
        tip_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        tip_pos = self._data.site_xpos[tip_id][:2]
        dist = float(np.linalg.norm(tip_pos - self._target_pos))
        tactile = np.array([float(dist <= 0.05)], dtype=np.float32)
        return SensoryBundle(
            proprioception=proprio,
            tactile=tactile,
            timestamp_ms=self._t_ms,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
