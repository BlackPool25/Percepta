"""Custom MuJoCo navigation environment.

Open arena with:
  - Agent sphere (blue) with 2D force control
  - Goal zone (green cylinder) at fixed position
  - 2 colored boxes (decorative, for future pushing tasks)
  - Full state observation + rendered image

Observation (12-dim state + 3×64×64 image):
  [agent_x, agent_y, agent_vx, agent_vy,
   goal_x, goal_y,
   obj1_x, obj1_y, obj2_x, obj2_y,
   contact_obj1, contact_obj2]

Action (2-dim): force_x, force_y in [-1, 1]

Reward: -0.1 * ||agent - goal|| + 100.0 if within goal_radius

Curriculum:
  Phase 1: Fixed start + fixed goal (seed-based)
  Phase 2: Random start + fixed goal
  Phase 3: Random start + random goal
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from typing import Optional

XML = r"""
<mujoco model="nav_arena">
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

    <!-- Boundary walls -->
    <geom name="wall_nx" type="box" size="0.05 5 0.5" pos="-5 0 0.25" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_px" type="box" size="0.05 5 0.5" pos="5 0 0.25" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_ny" type="box" size="5 0.05 0.5" pos="0 -5 0.25" rgba="0.4 0.4 0.4 1"/>
    <geom name="wall_py" type="box" size="5 0.05 0.5" pos="0 5 0.25" rgba="0.4 0.4 0.4 1"/>

    <!-- Movable maze walls (position randomized per episode) -->
    <geom name="maze1" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze2" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze3" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze4" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze5" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze6" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze7" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze8" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze9" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze10" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze11" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze12" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze13" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze14" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze15" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze16" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze17" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze18" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze19" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>
    <geom name="maze20" type="box" size="0.05 1.5 0.25" pos="0 0 0.25" rgba="0.6 0.3 0.1 1"/>

    <!-- Goal zone: tall green cylinder -->
    <geom name="goal" type="cylinder" size="0.3 0.02" pos="3 3 0.01" rgba="0.0 0.9 0.0 0.6"/>

    <!-- Agent sphere -->
    <body name="agent" pos="0 0 0.5">
      <freejoint name="agent_joint"/>
      <geom name="agent_geom" type="sphere" size="0.3" rgba="0.2 0.6 1.0 1" mass="1.0"/>
    </body>

    <!-- Decorative objects (visible but not needed for navigation) -->
    <body name="obj1" pos="-2 -2 0.25">
      <freejoint name="obj1_joint"/>
      <geom name="obj1_geom" type="box" size="0.2 0.2 0.2" rgba="1.0 0.3 0.3 1" mass="0.5"/>
    </body>

    <body name="obj2" pos="2 -2 0.35">
      <freejoint name="obj2_joint"/>
      <geom name="obj2_geom" type="box" size="0.3 0.3 0.3" rgba="0.3 1.0 0.3 1" mass="2.0"/>
    </body>

    <!-- Goal marker pole (visual indicator) — positioned directly in reset -->
  </worldbody>

</mujoco>
"""


class NavArena(gym.Env):
    """Open navigation arena with visible goal and objects."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 200,
                 force_scale: float = 30.0, goal_radius: float = 0.5):
        super().__init__()
        self.max_steps = max_steps
        self.force_scale = force_scale
        self.goal_radius = goal_radius

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._agent_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
        self._obj_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
            for i in range(1, 3)
        ]
        self._agent_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "agent_geom")
        self._obj_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obj{i}_geom")
            for i in range(1, 3)
        ]
        self._maze_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"maze{i}")
            for i in range(1, 21)
        ]
        # Fixed goal position
        self._goal_pos = np.array([3.0, 3.0, 0.0], dtype=np.float32)

        # Maze wall configurations: list of (geom_id, x, y) positions
        self._wall_configs = [
            [(0, -2.0), (0, 0.0), (1.5, 1.5), (-1.5, -1.5)],  # config 1
            [(1.5, 0.0), (-1.5, 0.0), (0, 1.5), (0, -1.5)],   # config 2
            [(2.0, 0.0), (-2.0, 0.0), (0, 2.0), (0, -2.0)],   # config 3
            [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)],  # config 4
        ]
        self._random_wall_mode = False  # Phase 4: truly random walls

        # Observation: 12-dim state + rendered image
        self._state_dim = 12
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32),
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Rendering
        self.render_mode = render_mode
        self._renderer = None
        if render_mode in ("rgb_array", "human"):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.step_count = 0
        self._fixed_start = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        self._start_randomize = False
        self._randomize_maze = False

    def set_curriculum(self, phase: int, rng=None):
        """Set curriculum phase:
        0: Fixed start + fixed goal (easiest)
        1: Random start + fixed goal
        2: Random start + random goal
        3: Random start + random goal + 4 pre-defined wall configs
        4: Random start + random goal + TRULY RANDOM MAZES (8-12 walls)
        """
        self._start_randomize = phase >= 1
        self._randomize_maze = phase >= 3 and phase < 4
        self._random_wall_mode = phase >= 4
        if phase >= 2:
            rng = rng or self.np_random
            self._goal_pos = np.array(
                [rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0],
                dtype=np.float32
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize maze walls
        if self._random_wall_mode:
            # Phase 4: MASSIVE MAZES — 18-20 walls forming corridor structures
            n_walls = self.np_random.integers(18, 21)
            placements = []
            # Generate walls in a structured maze pattern
            for i in range(min(n_walls, len(self._maze_geom_ids))):
                # Create corridors by placing walls in rows/columns
                if self.np_random.random() < 0.5:
                    # Vertical wall: blocks a y-range
                    x = self.np_random.uniform(-4.0, 4.0)
                    y = self.np_random.uniform(-3.0, 3.0)
                    placements.append((x, y, 0))  # 0=vertical
                else:
                    # Horizontal wall: blocks an x-range
                    x = self.np_random.uniform(-3.0, 3.0)
                    y = self.np_random.uniform(-4.0, 4.0)
                    placements.append((x, y, 1))  # 1=horizontal
            for i, geom_id in enumerate(self._maze_geom_ids[:len(placements)]):
                x, y, orient = placements[i]
                if orient == 0:  # vertical wall
                    self.model.geom_pos[geom_id] = [x, y, 0.25]
                    self.model.geom_size[geom_id] = [0.05, 1.5, 0.25]
                else:  # horizontal wall
                    self.model.geom_pos[geom_id] = [x, y, 0.25]
                    self.model.geom_size[geom_id] = [1.5, 0.05, 0.25]
            for i in range(len(placements), len(self._maze_geom_ids)):
                self.model.geom_pos[self._maze_geom_ids[i]] = [100, 100, 0.25]
        elif self._randomize_maze:
            config = self._wall_configs[self.np_random.integers(0, len(self._wall_configs))]
            for i, geom_id in enumerate(self._maze_geom_ids[:len(config)]):
                x, y = config[i]
                self.model.geom_pos[geom_id] = [x, y, 0.25]
            # Hide unused maze geoms
            for i in range(len(config), len(self._maze_geom_ids)):
                self.model.geom_pos[self._maze_geom_ids[i]] = [100, 100, 0.25]
        else:
            # Hide maze walls (move outside arena)
            for geom_id in self._maze_geom_ids:
                self.model.geom_pos[geom_id] = [100, 100, 0.25]

        # Set agent position
        if self._start_randomize:
            box = 3.5
            eps = 0.3
            start_pos = np.array([
                self.np_random.uniform(-box + eps, box - eps),
                self.np_random.uniform(-box + eps, box - eps),
                0.5
            ], dtype=np.float32)
        else:
            start_pos = self._fixed_start.copy()

        self._set_body_pos("agent", start_pos)

        # Set goal position
        goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
        self.model.geom_pos[goal_geom_id] = self._goal_pos.copy()

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def _set_body_pos(self, body_name: str, pos: np.ndarray):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.data.qpos[qpos_adr:qpos_adr + 3] = pos[:3]
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    def step(self, action: np.ndarray):
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

        truncated = self.step_count >= self.max_steps

        return obs, float(reward), terminated, truncated, {}

    def _get_obs(self) -> dict:
        agent_pos = self.data.xpos[self._agent_body_id].copy().astype(np.float32)
        agent_vel = self.data.cvel[self._agent_body_id, 3:6].copy().astype(np.float32)

        obj_positions = []
        for bid in self._obj_body_ids:
            obj_positions.append(self.data.xpos[bid, :2].copy().astype(np.float32))

        contacts = [0.0, 0.0]
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

        state = np.concatenate([
            agent_pos[:2],          # 0:2 — agent x, y
            agent_vel[:2],          # 2:4 — agent vx, vy
            obj_positions[0],       # 4:6 — obj1 x, y
            obj_positions[1],       # 6:8 — obj2 x, y
            contacts,               # 8:10 — contact flags
        ]).astype(np.float32)

        image = self.render()
        if image is None:
            image = np.zeros((64, 64, 3), dtype=np.uint8)

        return {'state': state, 'image': image}

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            img = self._renderer.render()
            # Resize to 64×64 for the autoencoder
            from PIL import Image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((64, 64), Image.LANCZOS)
            return np.array(pil_img, dtype=np.uint8)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
