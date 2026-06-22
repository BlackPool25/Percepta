"""Multi-Rule Arena: tests context-dependent rule learning.
Two zones with DIFFERENT goal patterns:
  LEFT zone  (x < 0): goal always in NW area (x < -1, y > 1)
  RIGHT zone (x > 0): goal always in SE area (x > 1, y < -1)

Agent must learn WHICH rule applies based on its current position.
No explicit zone indicator — must infer from x-coordinate.
"""

import mujoco
import numpy as np
from gymnasium import spaces
from typing import Optional

XML = r"""
<mujoco model="bizonal_arena">
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
    <geom friction="0.1 0.005 0.001" solimp="0.95 0.99 0.01" solref="0.02 1"/>
  </default>
  <worldbody>
    <light name="light" pos="0 0 15" directional="true" castshadow="false"/>
    <geom name="floor" type="plane" size="5 5 0.02" material="floor"/>
    <geom name="wall_nx" type="box" size="0.05 5 0.5" pos="-5 0 0.25" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_px" type="box" size="0.05 5 0.5" pos="5 0 0.25" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_ny" type="box" size="5 0.05 0.5" pos="0 -5 0.25" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_py" type="box" size="5 0.05 0.5" pos="0 5 0.25" rgba="0.4 0.4 0.4 1"/>
    <geom name="goal" type="cylinder" size="0.3 0.02" pos="0 0 0.01" rgba="0.0 0.9 0.0 0.6"/>
    <body name="agent" pos="0 0 0.5">
      <freejoint name="agent_joint"/>
      <geom name="agent_geom" type="sphere" size="0.3" rgba="0.2 0.6 1.0 1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


class BizonalArena:
    """Two-zone arena with context-dependent goal rules.
    
    LEFT zone  (start.x < 0): goal in NW quadrant (x: -3 to -1, y: 1 to 3)
    RIGHT zone (start.x > 0): goal in SE quadrant (x: 1 to 3, y: -3 to -1)
    
    State (4-dim): [agent_x, agent_y, agent_vx, agent_vy]
    Action (2-dim): force_x, force_y in [-1, 1]
    Reward: -0.1*dist_to_goal + 100 if within goal_radius
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode=None, max_steps=200, force_scale=30.0, goal_radius=0.5):
        self.max_steps = max_steps
        self.force_scale = force_scale
        self.goal_radius = goal_radius
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self._agent_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self._renderer = None
        if render_mode in ("rgb_array", "human"):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _sample_goal(self, start_x):
        """Sample goal following zone rule based on start position."""
        if start_x < 0:
            # LEFT zone → goal in NW quadrant
            x = np.random.uniform(-3.0, -1.0)
            y = np.random.uniform(1.0, 3.0)
        else:
            # RIGHT zone → goal in SE quadrant  
            x = np.random.uniform(1.0, 3.0)
            y = np.random.uniform(-3.0, -1.0)
        return np.array([x, y, 0.0], dtype=np.float32)

    def reset(self, seed=None):
        mu = np.random.RandomState(seed) if seed is not None else np.random
        mujoco.mj_resetData(self.model, self.data)

        # Random start position
        start_x = mu.uniform(-3.5, 3.5)
        start_y = mu.uniform(-3.5, 3.5)
        self._set_body_pos("agent", np.array([start_x, start_y, 0.5]))

        # Goal follows zone rule based on start position
        self._goal_pos = self._sample_goal(start_x)
        goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
        self.model.geom_pos[goal_id] = self._goal_pos.copy()

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def _set_body_pos(self, body_name, pos):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.data.qpos[qpos_adr:qpos_adr + 3] = pos[:3]

    def step(self, action):
        force = np.clip(action, -1.0, 1.0) * self.force_scale
        for _ in range(4):
            self.data.xfrc_applied[self._agent_body_id, :2] = force[:2]
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        agent_pos = self.data.xpos[self._agent_body_id]
        dist = np.linalg.norm(agent_pos[:2] - self._goal_pos[:2])
        reward = -0.1 * dist
        terminated = False
        if dist < self.goal_radius:
            reward += 100.0
            terminated = True

        return obs, float(reward), terminated, self.step_count >= self.max_steps, {}

    def _get_obs(self):
        """10-dim state matching original arena for shared policy use.
        [x, y, vx, vy, 0, 0, 0, 0, 0, 0] — extra dims zero (no objects).
        """
        agent_pos = self.data.xpos[self._agent_body_id].copy().astype(np.float32)
        agent_vel = self.data.cvel[self._agent_body_id, 3:6].copy().astype(np.float32)
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = agent_pos[0]; obs[1] = agent_pos[1]
        obs[2] = agent_vel[0]; obs[3] = agent_vel[1]
        return obs

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
