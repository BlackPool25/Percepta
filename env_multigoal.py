"""Multi-goal MuJoCo arena: requires hierarchical planning / DLPFC.

The agent must navigate to a WAYPOINT first, then to the GOAL.
The goal is LOCKED until the waypoint is reached.
This forces multi-step planning — the DLPFC must maintain the
sequence "waypoint → goal" in working memory.

Observation (13-dim): [agent_x, agent_y, goal_x, goal_y, agent_vx, agent_vy,
                       waypoint_x, waypoint_y, obj1_x, obj1_y, obj2_x, obj2_y, contacts, 
                       waypoint_reached (0/1)]

Action (2-dim): force_x, force_y in [-1, 1]

Reward:
  -0.1 * dist_to_waypoint (if waypoint not reached)
  -0.1 * dist_to_goal + 100 (if waypoint reached and goal reached)
"""

import numpy as np
from env_nav import NavArena


class MultiGoalArena(NavArena):
    """Navigation arena with sequential waypoint → goal requirement.

    Extends NavArena with a waypoint that must be reached before
    the goal unlocks. Tests DLPFC multi-step planning.
    """

    def __init__(self, render_mode=None, max_steps=400):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self._state_dim = 14  # 12 base + waypoint_reached flag + waypoint pos
        self._waypoint_pos = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        self.waypoint_reached = False
        self.waypoint_radius = 0.7

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.waypoint_reached = False
        # Randomize waypoint position
        rng = self.np_random
        self._waypoint_pos = np.array(
            [rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0],
            dtype=np.float32
        )
        # Ensure waypoint isn't too close to goal
        while np.linalg.norm(self._waypoint_pos[:2] - self._goal_pos[:2]) < 1.5:
            self._waypoint_pos = np.array(
                [rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0],
                dtype=np.float32
            )
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        agent_pos = self.data.xpos[self._agent_body_id]

        # Check waypoint proximity
        if not self.waypoint_reached:
            dist_to_wp = np.linalg.norm(agent_pos[:2] - self._waypoint_pos[:2])
            if dist_to_wp < self.waypoint_radius:
                self.waypoint_reached = True

        # Recompute reward based on waypoint status
        dist_to_goal = np.linalg.norm(agent_pos[:2] - self._goal_pos[:2])
        if self.waypoint_reached:
            reward = -0.1 * dist_to_goal
            if dist_to_goal < self.goal_radius:
                reward += 100.0
                terminated = True
        else:
            dist_to_wp = np.linalg.norm(agent_pos[:2] - self._waypoint_pos[:2])
            reward = -0.1 * dist_to_wp

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        base_obs = super()._get_obs()
        state = base_obs['state']
        # Append waypoint status and waypoint position
        extra = np.array([
            float(self.waypoint_reached),
            self._waypoint_pos[0],
            self._waypoint_pos[1],
        ], dtype=np.float32)
        new_state = np.concatenate([state, extra])
        base_obs['state'] = new_state
        return base_obs

    def set_curriculum(self, phase, rng=None):
        """Phase 5: multi-goal task."""
        super().set_curriculum(phase, rng)
        # Phase 5 = multi-goal
        if phase >= 5:
            self._start_randomize = True
