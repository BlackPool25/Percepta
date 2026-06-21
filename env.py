"""MuJoCo 3D Playground — agent + objects + goal in a physics world.

Observation (34-dim):
  agent_pos (3), agent_vel (3)
  obj1_pos (3), obj1_vel (3)
  obj2_pos (3), obj2_vel (3)
  obj3_pos (3), obj3_vel (3)
  goal_pos (3)
  contacts (3 binary: agent-obj1/2/3)
  task (3 one-hot: which object to push)

Action (3-dim): force_x, force_y, force_z in [-1, 1]
"""

import math
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


XML = r"""
<mujoco model="playground">
  <compiler angle="degree" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <global azimuth="135" elevation="-30"/>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.15 0.20 0.25" rgb2="0.10 0.15 0.20"/>
    <material name="floor" texture="grid" texrepeat="3 3" reflectance="0.1"/>
  </asset>

  <default>
    <geom friction="0.2 0.01 0.001" solimp="0.99 0.99 0.01" solref="0.01 1"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 15" directional="true" castshadow="false"/>
    <geom name="floor" type="plane" size="5 5 0.02" material="floor"/>

    <!-- Walls -->
    <geom name="wall_nx" type="box" size="0.1 5 0.5" pos="-5 0 0.5" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_px" type="box" size="0.1 5 0.5" pos="5 0 0.5" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_ny" type="box" size="5 0.1 0.5" pos="0 -5 0.5" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_py" type="box" size="5 0.1 0.5" pos="0 5 0.5" rgba="0.4 0.4 0.4 1"/>

    <!-- Goal zone -->
    <geom name="goal" type="cylinder" size="0.5 0.02" pos="3 3 0.01" rgba="0.2 0.9 0.2 0.5"/>

    <!-- Agent sphere -->
    <body name="agent" pos="0 0 0.5">
      <freejoint name="agent_joint"/>
      <geom name="agent_geom" type="sphere" size="0.3" rgba="0.2 0.6 1.0 1" mass="1.0"/>
    </body>

    <!-- Object 1: small box, light -->
    <body name="obj1" pos="-2 -2 0.25">
      <freejoint name="obj1_joint"/>
      <geom name="obj1_geom" type="box" size="0.2 0.2 0.2" rgba="1.0 0.3 0.3 1" mass="0.5"/>
    </body>

    <!-- Object 2: medium box, heavy -->
    <body name="obj2" pos="2 -2 0.35">
      <freejoint name="obj2_joint"/>
      <geom name="obj2_geom" type="box" size="0.3 0.3 0.3" rgba="0.3 1.0 0.3 1" mass="2.0"/>
    </body>

    <!-- Object 3: sphere, light -->
    <body name="obj3" pos="0 -3 0.3">
      <freejoint name="obj3_joint"/>
      <geom name="obj3_geom" type="sphere" size="0.25" rgba="1.0 0.8 0.2 1" mass="0.25"/>
    </body>
  </worldbody>
</mujoco>
"""


class MuJoCoPlayground(gym.Env):
    """3D physics playground with a spherical agent and pushable objects."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 500,
                 force_scale: float = 15.0, goal_radius: float = 0.6,
                 curriculum_dist: float = 4.5):
        """curriculum_dist: initial distance from goal to target object spawn.
        Decrease over episodes to make the task easier."""
        super().__init__()
        self.max_steps = max_steps
        self.force_scale = force_scale
        self.goal_radius = goal_radius
        self.curriculum_dist = curriculum_dist

        # Build MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        # Cache IDs for fast access
        self._agent_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
        self._obj_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
            for i in range(1, 4)
        ]
        self._agent_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "agent_geom")
        self._obj_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obj{i}_geom")
            for i in range(1, 4)
        ]
        self._goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal")

        # Goal position (fixed)
        self._goal_pos = np.array([3.0, 3.0, 0.0], dtype=np.float32)

        # Spawn positions for randomization
        self._obj_spawns = [
            np.array([-2.0, -2.0, 0.25]),
            np.array([2.0, -2.0, 0.35]),
            np.array([0.0, -3.0, 0.3]),
        ]
        self._agent_spawn = np.array([0.0, 0.0, 0.5])

        # Observation space
        # agent(6) + 3 × objects(6 each) + goal(3) + contacts(3) + task(3)
        self._obs_dim = 6 + 18 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        # Action space: 3D force on agent body
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Rendering
        self.render_mode = render_mode
        self._renderer = None
        if render_mode in ("rgb_array", "human"):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        # State
        self.step_count = 0
        self._target_object = 0
        self._task_done = False
        self._last_action = np.zeros(3, dtype=np.float32)
        self._prev_dist_to_goal = 7.0  # initial distance estimate
        self._best_dist_to_goal = 7.0  # personal best (for HER-style reward)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._randomize_positions()
        self._target_object = self.np_random.integers(0, 3)

        # Place goal at curriculum distance from target object's ACTUAL position
        obj_id = self._obj_body_ids[self._target_object]
        jnt_adr = self.model.body_jntadr[obj_id]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        target_pos_actual = self.data.qpos[qpos_adr:qpos_adr + 3].copy()
        angle = self.np_random.uniform(0, 2 * np.pi)
        goal_offset = np.array([
            self.curriculum_dist * np.cos(angle),
            self.curriculum_dist * np.sin(angle),
            0.0
        ])
        self._goal_pos = target_pos_actual + goal_offset
        self._goal_pos[2] = 0.0  # on the floor

        # Update goal geom position (visual)
        goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
        self.model.geom_pos[goal_geom_id] = self._goal_pos.copy()
        self._task_done = False
        self.step_count = 0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._prev_dist_to_goal = 7.0
        self._best_dist_to_goal = 7.0

        # Update positions in MuJoCo
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _randomize_positions(self):
        """Randomize agent and object start positions within bounds."""
        eps = 0.5  # margin from walls
        box = 4.0  # half-size of spawn area
        ag = self._agent_spawn.copy()
        ag[:2] = self.np_random.uniform(low=-box + eps, high=box - eps, size=2)
        self._set_body_pos("agent", ag)

        for i in range(3):
            pos = self._obj_spawns[i].copy()
            pos[:2] = self.np_random.uniform(low=-box + eps, high=box - eps, size=2)
            self._set_body_pos(f"obj{i+1}", pos)

    def _set_body_pos(self, body_name: str, pos: np.ndarray):
        """Set the position of a body with a free joint.
        Free joint qpos layout: [x, y, z, qw, qx, qy, qz]
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        # qpos[qpos_adr:qpos_adr+3] = position
        # qpos[qpos_adr+3:qpos_adr+7] = quaternion
        self.data.qpos[qpos_adr:qpos_adr + 3] = pos[:3]
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]  # identity

    def step(self, action: np.ndarray):
        self._last_action = np.asarray(action, dtype=np.float32).copy()

        # Apply force to agent body
        force = np.clip(action, -1.0, 1.0) * self.force_scale
        self.data.xfrc_applied[self._agent_body_id, :3] = force

        # Step physics (multiple sub-steps for stability)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()

        # Clear applied forces for next step
        self.data.xfrc_applied[self._agent_body_id, :] = 0.0

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        parts = []

        # Agent position (world) and linear velocity
        parts.append(self.data.xpos[self._agent_body_id].copy().astype(np.float32))
        parts.append(self.data.cvel[self._agent_body_id, 3:6].copy().astype(np.float32))

        # Object positions and velocities
        for bid in self._obj_body_ids:
            parts.append(self.data.xpos[bid].copy().astype(np.float32))
            parts.append(self.data.cvel[bid, 3:6].copy().astype(np.float32))

        # Goal position
        parts.append(self._goal_pos.astype(np.float32))

        # Contact indicators
        contacts = self._detect_contacts()
        parts.append(np.array(contacts, dtype=np.float32))

        # Task encoding (one-hot: which object to push)
        task = np.zeros(3, dtype=np.float32)
        task[self._target_object] = 1.0
        parts.append(task)

        return np.concatenate(parts)

    def _detect_contacts(self) -> list[float]:
        """Return binary contact flags for agent with each object."""
        contacts = [0.0, 0.0, 0.0]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._agent_geom_id:
                for j, og in enumerate(self._obj_geom_ids):
                    if g2 == og:
                        contacts[j] = 1.0
            elif g2 == self._agent_geom_id:
                for j, og in enumerate(self._obj_geom_ids):
                    if g1 == og:
                        contacts[j] = 1.0
        return contacts

    def _compute_reward(self) -> float:
        target_id = self._obj_body_ids[self._target_object]
        target_pos = self.data.xpos[target_id]
        agent_pos = self.data.xpos[self._agent_body_id]
        goal_pos = self._goal_pos

        dist_tg = np.linalg.norm(target_pos[:2] - goal_pos[:2])
        dist_at = np.linalg.norm(agent_pos[:2] - target_pos[:2])

        reward = 0.0

        # ─── Intrinsic motivation: approach target object ───
        # Gentle gradient: reward for being close to the target object
        reward += 0.05 * np.exp(-0.5 * dist_at)

        # ─── Exponential proximity reward to goal ───
        # Dense gradient when target is near goal
        reward += 20.0 * np.exp(-1.0 * dist_tg)

        # ─── HER personal best (exponential improvement bonus) ───
        if dist_tg < self._best_dist_to_goal:
            old_prox = 1.0 / (self._best_dist_to_goal + 0.1)
            new_prox = 1.0 / (dist_tg + 0.1)
            improvement = new_prox - old_prox
            reward += 3.0 * improvement
            self._best_dist_to_goal = dist_tg

        # ─── Goal completion: massive reward spike ───
        if dist_tg < self.goal_radius:
            reward += 100.0
            self._task_done = True

        self._prev_dist_to_goal = dist_tg

        # Weak distance penalty
        reward -= 0.001 * dist_tg

        # Action penalty
        reward -= 0.001 * np.sum(self._last_action ** 2)

        # Time penalty: encourages efficient goal-reaching
        reward -= 0.01

        return float(reward)

    def _check_termination(self) -> bool:
        return self._task_done

    def _get_info(self) -> dict:
        return {
            "target_object": int(self._target_object),
            "task_done": bool(self._task_done),
            "step": self.step_count,
        }

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
