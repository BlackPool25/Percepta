"""Step 10: Embodied Learning — NavArena with Episodic Control + Curriculum.

Architecture (research-backed):
  1. NavArena: open world with agent, goal, objects (12-dim state + 64×64 image)
  2. DG (PatternSeparator) on agent position → sparse position keys
  3. ImportanceWeightedMemory: stores (position_key, state, action, Q_value)
  4. Episodic Control: retrieve Q_star from memory, use if Q_star > V_policy
  5. CA1 novelty: position distance to nearest stored state
  6. Reverse replay: propagate Q-values backward after successful episodes
  7. Curriculum: fixed goal → random start → random goal
  8. Sleep: autoencoder consolidation (optional, for visual features)

Non-negotiables satisfied:
  R1: Fast (HopfieldMemory) + Slow (Autoencoder) — one-shot + gradient
  R2: DG pattern separation — sparse discriminative position keys
  R4: Sleep consolidation + reverse replay — credit assignment
  R5: Importance-weighted retention — STC-like memory management
  R6: CA1 novelty gates storage and intrinsic reward
  R7: Key-value Hopfield retrieval — content-addressable memory
  R8: Single-pass streaming — online learning without IID assumption
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from pathlib import Path
from PIL import Image

from hopfield_memory import PatternSeparator
from env_nav import NavArena

OUT = Path('results/step10')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════
# VICReg — Prevents feature collapse (for optional autoencoder)
# ═══════════════════════════════════════════════════════════════════

def vicreg_loss(h, var_weight=1.0, cov_weight=0.04):
    std = torch.sqrt(h.var(dim=0) + 1e-4)
    var_loss = torch.mean(torch.clamp(1 - std, min=0))
    h_c = h - h.mean(dim=0)
    cov = (h_c.T @ h_c) / (h.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / h.size(1)
    return var_weight * var_loss + cov_weight * cov_loss


# ═══════════════════════════════════════════════════════════════════
# SUCCESSOR FEATURE NETWORK (SR — allocentric predictive map)
# ═══════════════════════════════════════════════════════════════════
#
# Q(s) = φ(s)^T · w
# φ(s) = successor features (predictive map — environment dynamics)
# w = reward weights (the current goal)
#
# When the goal changes, only w is updated via TD.
# φ(s) stays the same — it captures "where I can go from here."

class SuccessorFeatureNet(nn.Module):
    """Maps state vector → successor features φ(s).
    φ(s) captures the expected future state occupancy (predictive map).
    """
    def __init__(self, state_dim=12, phi_dim=64):
        super().__init__()
        self.phi_dim = phi_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, phi_dim),
        )
        # Reward weight w — the only thing that changes when goal moves
        self.w = nn.Parameter(torch.zeros(phi_dim))

    def forward(self, state):
        """Returns φ(s). Q(s) = φ(s)^T · w computed by caller."""
        return self.net(state)

    def q_value(self, phi):
        """Compute Q from successor features: Q(s) = φ(s)^T · w."""
        return (phi * self.w.unsqueeze(0)).sum(dim=-1)

    def q_from_state(self, state):
        """Convenience: Q(s) = φ_net(s)^T · w."""
        phi = self.forward(state)
        return self.q_value(phi)

    def successor_loss(self, phi_s, phi_s_next, gamma=0.99):
        """TD loss on successor features:
        ψ(s) should predict φ(s) + γ·ψ(s')
        But for simplicity: φ(s) should predict φ(s) + γ·φ(s')
        """
        target = phi_s + gamma * phi_s_next.detach()
        return F.mse_loss(phi_s, target)


# ═══════════════════════════════════════════════════════════════════
# COMPONENT: Importance-Weighted Memory (CA3 + importance tracking)
# ═══════════════════════════════════════════════════════════════════
#
# Maps to brain (R5: Synaptic Tagging and Capture):
#   - Storage: every pattern gets an initial "tag" (importance = 1.0)
#   - Retrieval: matched patterns get tag reinforcement (importance += attention_weight)
#   - Decay: all tags slowly weaken (decay ≈ 0.999 per step)
#   - Eviction: weakest tag is pruned (active forgetting of irrelevant memories)
#   - Sleep: replayed proportional to tag strength² (PRP allocation ∝ importance)

class ImportanceWeightedMemory:
    """KeyValueHopfieldMemory with episodic action retrieval.

    Keys: DG-separated sparse codes from STATE (position)
    Values: stored STATE + visual + ACTION + RETURN
    Stores full episodic traces: where I was, what I saw, what I did, what happened.

    This enables one-shot learning: when the agent revisits a state,
    it retrieves not just familiarity but also the action that was taken
    and the outcome. The policy uses this to repeat successful actions.

    Importance tracks retrieval frequency (Synaptic Tagging and Capture).
    At capacity, the LEAST important pattern is evicted (not oldest).
    """

    def __init__(self, key_dim=2000, state_dim=12, phi_dim=64,
                 beta=2.0, max_size=500):
        self.keys = []          # DG-separated sparse position keys
        self.states = []        # full state vector (12-dim)
        self.phsis = []         # successor features φ(s) — allocentric predictive map
        self.importance = []    # bounded [0.01, 1.0]
        self.max_size = max_size
        self.beta = beta
        self.state_dim = state_dim
        self.phi_dim = phi_dim
        self.last_retrieved_idx = None
        self.step = 0
        self.retrieval_count = 0

    def store(self, key, state, phi):
        importances = np.array(self.importance) if self.importance else np.array([])
        if len(self.keys) >= self.max_size:
            idx = int(np.argmin(importances))
            self._evict(idx)
        self.keys.append(key.cpu().detach())
        self.states.append(state.cpu().detach())
        self.phsis.append(phi.cpu().detach())
        self.importance.append(0.5)

    def retrieve(self, query_key, sf_net=None):
        """Retrieve by position key.
        Returns: (state_star, phi_star, rbf_sim)
        Q(s) = φ(s)^T · w is computed by the caller using sf_net.w
        """
        if not self.keys:
            self.last_retrieved_idx = None
            return (torch.zeros(1, self.state_dim, device=query_key.device),
                    torch.zeros(1, self.phi_dim, device=query_key.device), 0.0)

        Z = torch.stack(self.keys, dim=0).to(device=query_key.device, dtype=query_key.dtype)
        S = torch.stack(self.states, dim=0).to(device=query_key.device, dtype=query_key.dtype)
        P = torch.stack(self.phsis, dim=0).to(device=query_key.device, dtype=query_key.dtype)

        logits = (query_key @ Z.T) * self.beta
        attn = F.softmax(logits, dim=-1)

        query_norm = query_key / (query_key.norm(dim=-1, keepdim=True) + 1e-8)
        Z_norm = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)
        cosims = (query_norm @ Z_norm.T).squeeze(0)
        rbf_sim = torch.exp(-2.0 * (1.0 - cosims)).max().item()

        best_idx = attn.argmax(dim=-1).item()
        attn_confidence = attn.max(dim=-1).values.item()

        state_star = attn @ S
        phi_star = attn @ P

        self.last_retrieved_idx = best_idx if attn_confidence > 0.1 else None
        self._update_importance(attn.squeeze(0))
        return state_star, phi_star, rbf_sim

    def get_q(self, idx, sf_net):
        """Compute Q(s) = φ(s)^T · w for pattern idx using current w."""
        if idx is None or idx >= len(self.phsis):
            return 0.0
        phi = self.phsis[idx].to(next(sf_net.parameters()).device)
        return (phi * sf_net.w).sum().item()

    def get_phi(self, idx):
        if idx is None or idx >= len(self.phsis):
            return None
        return self.phsis[idx]

    def _update_importance(self, attn_weights):
        self.retrieval_count += 1
        weights = attn_weights.detach().cpu().numpy()
        for i, w in enumerate(weights):
            if w > 0.05:
                self.importance[i] = min(self.importance[i] + 0.05, 1.0)

    def decay_importance(self, factor=0.99):
        for i in range(len(self.importance)):
            self.importance[i] = max(self.importance[i] * factor, 0.01)

    def sample_prioritized(self, batch_size, temperature=0.5):
        n = len(self.states)
        if n == 0:
            return None, None, None
        imp = np.array(self.importance, dtype=np.float64)
        imp = np.clip(imp, 0.01, 1.0)
        probs = imp ** (1.0 / temperature)
        probs = probs / (probs.sum() + 1e-10)
        idx = np.random.choice(n, min(batch_size, n), replace=False, p=probs)
        keys = torch.stack([self.keys[i] for i in idx])
        states = torch.stack([self.states[i] for i in idx])
        phis = torch.stack([self.phsis[i] for i in idx])
        return keys, states, phis

    def recompute_keys(self, sep):
        """Recompute position keys from stored states (first 2 dims = position)."""
        if not self.states:
            return
        with torch.no_grad():
            for i in range(len(self.states)):
                pos = self.states[i][:2].unsqueeze(0)
                z_i = sep(pos).squeeze(0)
                self.keys[i] = z_i.cpu()

    def _evict(self, idx):
        self.keys.pop(idx)
        self.states.pop(idx)
        self.phsis.pop(idx)
        self.importance.pop(idx)

    def stored_count(self):
        return len(self.keys)

    def __len__(self):
        return len(self.keys)

    def importance_stats(self):
        if not self.importance:
            return {}
        imp = np.array(self.importance)
        return {'mean': imp.mean(), 'std': imp.std(), 'min': imp.min(), 'max': imp.max()}


# ═══════════════════════════════════════════════════════════════════
# COMPONENT: Memory-Augmented Policy Head
# ═══════════════════════════════════════════════════════════════════
# Input = full state (12-dim) + retrieved action (2-dim) = 14-dim

class AugmentedPolicyHead(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        )
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.value = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x):
        h = self.net(x)
        mean = torch.tanh(self.policy_mean(h))
        std = F.softplus(self.log_std) + 1e-4
        value = self.value(h).squeeze(-1)
        return mean, std, value

    def act(self, x, deterministic=False):
        mean, std, value = self.forward(x)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, mean, std


# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

def make_env(render=False):
    return gym.make(
        'PointMaze_UMazeDense-v3',
        render_mode='human' if render else 'rgb_array',
        continuing_task=False,
    )


def process_obs(env, obs, info):
    """Render maze to 64×64 RGB tensor. Also return 2D position."""
    raw = env.render()
    img = Image.fromarray(raw).resize((64, 64))
    frame = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(frame).permute(2, 0, 1).float().to(DEVICE), obs['observation'][:2]


# ═══════════════════════════════════════════════════════════════════
# PPO: Generalized Advantage Estimation
# ═══════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


# ═══════════════════════════════════════════════════════════════════
# RUNNING NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

class RunningNorm:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.std = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if self.count < 100:
            return np.clip(x / (self.std + 1e-8), -5, 5)
        return np.clip((x - self.mean) / (self.std + 1e-8), -5, 5)

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if np.isscalar(x):
            x = np.array([x])
        batch_mean = x.mean()
        batch_std = x.std() + 1e-8
        n = len(x) if hasattr(x, '__len__') else 1
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        self.std = np.sqrt(self.std**2 + batch_std**2 * n / self.count +
                           delta**2 * n * (self.count - n) / self.count / self.count)


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

class Config:
    total_steps = 20000
    steps_per_rollout = 512
    ppo_epochs = 4
    minibatch_size = 64
    policy_lr = 3e-4
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    max_grad_norm = 0.5
    alpha_start = 1.0       # Initial: only intrinsic (exploration)
    alpha_end = 0.1         # Final: mostly extrinsic (exploitation)
    memory_size = 500       # Larger capacity for longer training
    trajectory_interval = 2000
    importance_decay = 0.999


def log_trajectory(positions, step, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions = np.array(positions)
    plt.figure(figsize=(6, 6))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    plt.scatter(positions[0, 0], positions[0, 1], c='green', s=50, label='start')
    plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=50, label='end')
    plt.xlim(-0.5, 5.5)
    plt.ylim(-0.5, 5.5)
    plt.gca().set_aspect('equal')
    plt.title(f'Trajectory at step {step}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path / f'trajectory_{step:06d}.png', dpi=100)
    plt.close()


def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("Percepta Step 9: Embodied Learning — Full Architecture")
    print("=" * 60)

    cfg = Config()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # 1. COMPONENTS
    # ══════════════════════════════════════════════════════════════
    state_dim = 12  # NavArena full state
    dg_pos = PatternSeparator(2, 2000, 0.02)  # agent position → sparse key
    memory = ImportanceWeightedMemory(key_dim=2000, state_dim=state_dim,
                                      beta=2.0, max_size=cfg.memory_size)
    # Policy input: full state (12) + retrieved goal_dir (2) = 14
    policy = AugmentedPolicyHead(input_dim=state_dim + 2, hidden_dim=64, action_dim=2).to(DEVICE)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)

    # Slow system (neocortex analogue) — learns Q-values from replayed patterns
    slow_net = SlowValueNet(state_dim=state_dim).to(DEVICE)
    slow_opt = torch.optim.Adam(slow_net.parameters(), lr=1e-3)

    # ══════════════════════════════════════════════════════════════
    # 2. LOGGING
    # ══════════════════════════════════════════════════════════════
    log = {
        'step': [], 'total_rew': [], 'ext_rew': [], 'ca1_nov': [],
        'policy_loss': [], 'value_loss': [], 'entropy': [], 'explained_var': [],
        'action_std': [], 'store_rate': [], 'mem_size': [],
        'imp_mean': [], 'imp_max': [], 'q_target_mean': [],
    }

    # ══════════════════════════════════════════════════════════════
    # 3. ENVIRONMENT (NavArena — fixed goal)
    # ══════════════════════════════════════════════════════════════
    env = NavArena(render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    state_vec = obs['state']  # 12-dim
    agent_pos = state_vec[:2]

    # ══════════════════════════════════════════════════════════════
    # 4. STATE TRACKING
    # ══════════════════════════════════════════════════════════════
    global_step = 0
    episode_rewards = []
    ep_reward_sum = 0
    ep_length = 0
    trajectory_positions = [agent_pos.copy()]
    stores_this_log = 0
    total_this_log = 0
    t_start = time.time()
    episodic_count = 0

    ca1_history = deque(maxlen=500)

    # ══════════════════════════════════════════════════════════════
    # 5. SEED MEMORY WITH DEMONSTRATION
    # ══════════════════════════════════════════════════════════════
    # Run a goal-directed heuristic to collect a successful trajectory.
    # Store each (state, action) in memory with high importance.
    # This gives the agent an initial "success memory" to retrieve.
    print("\nSeeding memory with demonstration trajectory...")
    demo_env = NavArena(render_mode='rgb_array')
    demo_success = False

    for demo_ep in range(5):
        demo_obs, _ = demo_env.reset(seed=42)
        demo_state = demo_obs['state']  # 12-dim
        demo_goal = demo_state[2:4]     # goal position from state
        demo_pos = demo_state[:2]
        demo_data = []

        for step_idx in range(500):
            delta = demo_goal - demo_pos
            dist = np.linalg.norm(delta)

            if dist > 1.0:
                # Strong goal-directed movement when far
                action = np.clip(delta / dist, -1, 1).astype(np.float32)
            elif dist > 0.3:
                # Slow approach when near
                action = np.clip(delta * 0.5, -0.3, 0.3).astype(np.float32)
            else:
                action = np.zeros(2, dtype=np.float32)

            obs2, reward, term, trunc, _ = demo_env.step(action)
            state2 = obs2['state']
            pos2 = state2[:2]

            demo_data.append({
                'state': demo_state.copy(),
                'action': action.copy(),
                'reward': reward,
            })

            demo_state, demo_pos = state2, pos2
            if term:
                demo_success = True
                break
            if trunc:
                break

        if demo_success:
            n = len(demo_data)
            qs = [0.0] * n
            cum = 0.0
            for i in range(n - 1, -1, -1):
                cum = demo_data[i]['reward'] + 0.99 * cum
                qs[i] = cum

            for i, d in enumerate(demo_data):
                s = d['state']
                goal_dir = (s[2:4] - s[:2]) / (np.linalg.norm(s[2:4] - s[:2]) + 1e-8)
                pos_t = torch.tensor(s[:2], dtype=torch.float32)
                z = dg_pos(pos_t.unsqueeze(0)).squeeze(0)
                state_t = torch.tensor(s, dtype=torch.float32)
                goal_dir_t = torch.tensor(goal_dir, dtype=torch.float32)
                memory.keys.append(z.cpu())
                memory.states.append(state_t)
                memory.goals.append(goal_dir_t)
                memory.q_values.append(qs[i])
                memory.importance.append(1.0)  # MAXIMUM importance — never evict

            print(f"  Demo {demo_ep + 1}: goal in {n} steps, {n} patterns stored, "
                  f"Q=[{qs[-1]:.1f}, {qs[0]:.1f}], importance=1.0 (protected)")
            break
        else:
            print(f"  Demo {demo_ep + 1}: did not reach goal — retrying...")

    demo_env.close()
    if not demo_success:
        print("  WARNING: No successful demo. Memory will be empty.")
    print(f"  Memory: {len(memory)} patterns")

    # ══════════════════════════════════════════════════════════════
    # MAIN LOOP (Curriculum: fixed start → random start → random goal)
    # ══════════════════════════════════════════════════════════════
    use_curriculum_seed = True
    consecutive_successes = 0
    episodic_count = 0
    sleep_counter = 0
    EPISODIC_Q_MARGIN = 0.05

    print(f"\nTraining ({cfg.total_steps} steps)...")
    print(f"  NavArena: open world, goal at (3, 3)")
    print(f"  Memory: {cfg.memory_size}, PPO epochs: {cfg.ppo_epochs}")
    print()

    while global_step < cfg.total_steps:
        # ─── ROLLOUT ────────────────────────────────────────────
        rollout = {k: [] for k in [
            'state', 'true_gd', 'action', 'log_prob', 'value', 'reward', 'done',
            'state_key', 'store_dec', 'ext_rew', 'ca1_nov', 'q_target',
        ]}

        goal_reached_here = False

        for _ in range(min(cfg.steps_per_rollout, cfg.total_steps - global_step)):
            with torch.no_grad():
                state_t = torch.from_numpy(state_vec).float().to(DEVICE).unsqueeze(0)
                pos_t = state_t[:, :2]
                goal_t = state_t[:, 2:4]

                # True goal direction (motor cortex uses actual goal location)
                true_goal_dir = (goal_t - pos_t) / ((goal_t - pos_t).norm(dim=-1, keepdim=True) + 1e-8)

                z = dg_pos(pos_t)

                # Memory retrieval — returns allocentric Q_value + RBF similarity
                state_star, q_star, rbf_sim = memory.retrieve(z)
                q_star_val = q_star.item()

                # CA1 novelty = position distance to nearest stored state
                ca1_nov = (pos_t - state_star[:, :2]).norm().item()

                # Policy input: state + TRUE goal direction
                pi_in = torch.cat([state_t.squeeze(0), true_goal_dir.squeeze(0)], dim=-1).unsqueeze(0)
                mean, std, value = policy(pi_in)
                dist = Normal(mean, std)
                policy_v = value.item()

                # Blended Q-target: RBF similarity is NOT diluted by pattern count.
                # When near a demo pattern (rbf_sim ≈ 1.0), Qtarget ≈ Q_demo (≈71).
                # When far from any pattern (rbf_sim ≈ 0), Qtarget ≈ V_policy.
                blend_weight = rbf_sim if rbf_sim > 0.05 else 0.0
                q_target_val = blend_weight * q_star_val + (1 - blend_weight) * policy_v

                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            obs2, reward_ext, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            done = term or trunc
            state_vec2 = obs2['state']

            # ─── TD UPDATE on the retrieved memory entry ───────────
            # Use next state's value as the TD target
            with torch.no_grad():
                next_state_t = torch.from_numpy(state_vec2).float().to(DEVICE).unsqueeze(0)
                next_goal_t = next_state_t[:, 2:4]
                next_pos_t = next_state_t[:, :2]
                next_goal_dir = (next_goal_t - next_pos_t) / ((next_goal_t - next_pos_t).norm(dim=-1, keepdim=True) + 1e-8)
                next_pi_in = torch.cat([next_state_t.squeeze(0), next_goal_dir.squeeze(0)], dim=-1).unsqueeze(0)
                _, _, next_value = policy(next_pi_in)

            td_error = memory.td_update(reward_ext, next_value.item(), gamma=cfg.gamma)
            memory.revise_on_mismatch(reward_ext, next_value.item(), gamma=cfg.gamma, mismatch_thresh=1.0)

            if term:
                consecutive_successes += 1
                goal_reached_here = True
            elif trunc:
                consecutive_successes = 0

            # Intrinsic: novelty annealed
            alpha_now = cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * \
                min(global_step / cfg.total_steps, 1.0)
            reward_int = alpha_now * ca1_nov
            total_reward = reward_ext + reward_int

            ca1_history.append(ca1_nov)

            rollout['state'].append(state_t.squeeze(0))
            rollout['true_gd'].append(true_goal_dir.squeeze(0))
            rollout['action'].append(action.squeeze(0))
            rollout['log_prob'].append(log_prob)
            rollout['value'].append(value)
            rollout['q_target'].append(q_target_val)
            rollout['reward'].append(total_reward)
            rollout['done'].append(1.0 if done else 0.0)
            rollout['state_key'].append(z.squeeze(0))
            store_dec = ca1_nov > np.percentile(list(ca1_history) + [0], 60) if len(ca1_history) > 20 else True
            rollout['store_dec'].append(store_dec)
            rollout['ext_rew'].append(reward_ext)
            rollout['ca1_nov'].append(ca1_nov)

            ep_reward_sum += reward_ext
            ep_length += 1
            global_step += 1
            trajectory_positions.append(state_vec[:2].copy())
            state_vec = state_vec2

            if done:
                episode_rewards.append(ep_reward_sum)
                consecutive_successes = consecutive_successes if term else 0
                if use_curriculum_seed:
                    obs_n, _ = env.reset(seed=42)
                else:
                    obs_n, _ = env.reset()
                state_vec = obs_n['state']
                ep_reward_sum = 0
                ep_length = 0

        # ─── CURRICULUM PHASE GATE ─────────────────────────────
        if use_curriculum_seed and consecutive_successes >= 5:
            print(f"  [curriculum] {consecutive_successes} successes → random start")
            use_curriculum_seed = False

        # ─── PPO UPDATE ────────────────────────────────────────
        state_batch = torch.stack(rollout['state'])
        action_batch = torch.stack(rollout['action'])
        old_lp_batch = torch.stack(rollout['log_prob'])
        reward_batch = torch.tensor(rollout['reward'], device=DEVICE)
        done_batch = torch.tensor(rollout['done'], device=DEVICE)
        value_batch = torch.stack(rollout['value'])

        # GAE bootstrap
        with torch.no_grad():
            state_n = torch.from_numpy(state_vec).float().to(DEVICE).unsqueeze(0)
            z_n = dg_pos(state_n[:, :2])
            s_star_n, a_star_n, _, _ = memory.retrieve(z_n)
            pi_n = torch.cat([state_n.squeeze(0), a_star_n.squeeze(0)], dim=-1).unsqueeze(0)
            _, _, next_value = policy(pi_n)

        all_v = torch.cat([value_batch.view(-1), next_value.view(-1)])
        adv, returns = compute_gae(reward_batch, all_v, done_batch, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = state_batch.size(0)
        ppo_log = {'pl': [], 'vl': [], 'ent': [], 'std': []}
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.minibatch_size):
                idx = perm[i:i + cfg.minibatch_size]

                mean, std, v_mb = policy(torch.cat([
                    state_batch[idx],
                    action_batch[idx]], dim=-1))
                dist = Normal(mean, std)
                lp = dist.log_prob(action_batch[idx]).sum(dim=-1)
                ent = dist.entropy().sum(dim=-1).mean()

                ratio = (lp - old_lp_batch[idx]).exp()
                ca = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[idx]
                pl = -(torch.min(ratio * adv[idx], ca)).mean()
                vl = F.mse_loss(v_mb, returns[idx])

                loss = pl + 0.5 * vl - cfg.entropy_coef * ent
                policy_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                policy_opt.step()

                ppo_log['pl'].append(pl.item())
                ppo_log['vl'].append(vl.item())
                ppo_log['ent'].append(ent.item())
                ppo_log['std'].append(std.mean().item())

        # ─── STORE EXPERIENCES IN MEMORY ───────────────────────
        for t in range(n):
            if rollout['store_dec'][t]:
                memory.store(
                    rollout['state_key'][t],
                    rollout['state'][t],
                    rollout['action'][t],
                    returns[t].item(),
                )
                stores_this_log += 1

        memory.decay_importance(factor=cfg.importance_decay)

        # ─── CURRICULUM: switch to random seeds after consistent success ───
        if use_curriculum_seed and consecutive_successes >= 5:
            print(f"  [curriculum] {consecutive_successes} consecutive successes → "
                  f"switching to random seeds")
            use_curriculum_seed = False
            consecutive_successes = 0

# ─── PPO UPDATE ────────────────────────────────────────
        state_batch = torch.stack(rollout['state'])
        true_gd_batch = torch.stack(rollout['true_gd'])
        action_batch = torch.stack(rollout['action'])
        old_lp_batch = torch.stack(rollout['log_prob'])
        reward_batch = torch.tensor(rollout['reward'], device=DEVICE, dtype=torch.float32)
        done_batch = torch.tensor(rollout['done'], device=DEVICE, dtype=torch.float32)
        value_batch = torch.stack(rollout['value'])
        q_target_batch = torch.tensor(rollout['q_target'], device=DEVICE, dtype=torch.float32)

        # GAE for policy gradient (uses actual rewards)
        with torch.no_grad():
            state_n = torch.from_numpy(state_vec).float().to(DEVICE).unsqueeze(0)
            pos_n = state_n[:, :2]
            goal_n = state_n[:, 2:4]
            true_gd_n = (goal_n - pos_n) / ((goal_n - pos_n).norm(dim=-1, keepdim=True) + 1e-8)
            z_n = dg_pos(pos_n)
            memory.retrieve(z_n)
            pi_n = torch.cat([state_n.squeeze(0), true_gd_n.squeeze(0)], dim=-1).unsqueeze(0)
            _, _, next_value = policy(pi_n)

        all_v = torch.cat([value_batch.view(-1), next_value.view(-1)])
        adv, returns = compute_gae(reward_batch, all_v, done_batch, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = state_batch.size(0)
        ppo_log = {'pl': [], 'vl': [], 'ent': [], 'std': []}
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.minibatch_size):
                idx = perm[i:i + cfg.minibatch_size]

                pi_input = torch.cat([state_batch[idx], true_gd_batch[idx]], dim=-1)
                mean, std, v_mb = policy(pi_input)
                dist = Normal(mean, std)
                lp = dist.log_prob(action_batch[idx]).sum(dim=-1)
                ent = dist.entropy().sum(dim=-1).mean()

                # Policy gradient: standard PPO clipped objective
                ratio = (lp - old_lp_batch[idx]).exp()
                ca = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[idx]
                pl = -(torch.min(ratio * adv[idx], ca)).mean()

                # Value loss: critic learns to predict BLENDED Q-target
                # (episodic Q + parametric V blended by confidence)
                vl = F.mse_loss(v_mb, q_target_batch[idx])

                loss = pl + 0.5 * vl - cfg.entropy_coef * ent
                policy_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                policy_opt.step()

                ppo_log['pl'].append(pl.item())
                ppo_log['vl'].append(vl.item())
                ppo_log['ent'].append(ent.item())
                ppo_log['std'].append(std.mean().item())

        # ─── STORE EXPERIENCES IN MEMORY ───────────────────────
        for t in range(n):
            if rollout['store_dec'][t]:
                memory.store(
                    rollout['state_key'][t],
                    rollout['state'][t],
                    q_target_batch[t].item(),
                )
                stores_this_log += 1

        # ─── SLEEP CONSOLIDATION (R4) ──────────────────────────
        sleep_counter += 1
        if sleep_counter % 4 == 0 and len(memory) >= cfg.minibatch_size:
            loss = sleep_consolidation(memory, slow_net, slow_opt, dg_pos, cfg)
            if loss > 0:
                print(f"  [sleep] slow net Q-MSE={loss:.6f}, mem={len(memory)}")

        # ─── LOGGING ───────────────────────────────────────────
        if global_step // cfg.steps_per_rollout > len(log['step']):
            log['step'].append(global_step)
            log['total_rew'].append(np.mean(rollout['reward']))
            log['ext_rew'].append(np.mean(rollout['ext_rew']))
            log['ca1_nov'].append(np.mean(rollout['ca1_nov']))
            n_actual = sum(rollout['store_dec'])
            log['store_rate'].append(stores_this_log / max(n_actual, 1))

            with torch.no_grad():
                _, _, vals = policy(torch.cat([state_batch, true_gd_batch], dim=-1))
                ev = 1 - ((vals - returns) ** 2).mean() / (returns.var() + 1e-8)
            log['policy_loss'].append(np.mean(ppo_log['pl']))
            log['value_loss'].append(np.mean(ppo_log['vl']))
            log['entropy'].append(np.mean(ppo_log['ent']))
            log['explained_var'].append(ev.item())
            log['action_std'].append(np.mean(ppo_log['std']))
            log['mem_size'].append(len(memory))
            q_t_mean = np.mean(rollout['q_target'])
            log['q_target_mean'].append(q_t_mean)

            imp_s = memory.importance_stats()
            log['imp_mean'].append(imp_s.get('mean', 0))
            log['imp_max'].append(imp_s.get('max', 0))

            elapsed = time.time() - t_start
            print(f"  step={global_step:6d} "
                  f"rew={np.mean(rollout['reward']):.3f} "
                  f"ext={np.mean(rollout['ext_rew']):.3f} "
                  f"ca1={log['ca1_nov'][-1]:.4f} "
                  f"store={log['store_rate'][-1]:.2f} "
                  f"mem={len(memory):3d} "
                  f"Qtarget={q_t_mean:.2f} "
                  f"act_std={log['action_std'][-1]:.3f} "
                  f"ev={log['explained_var'][-1]:.2f} "
                  f"t={elapsed:.0f}s")

            stores_this_log = 0

        if global_step % cfg.trajectory_interval == 0:
            log_trajectory(trajectory_positions, global_step, out)
            trajectory_positions = [state_vec[:2].copy()]

    # ─── FINAL ──────────────────────────────────────────────────
    torch.save(policy.state_dict(), out / 'policy.pt')
    print(f"\n  Saved: {out / 'policy.pt'}")
    elapsed = time.time() - t_start
    print(f"\n  Done: {global_step} steps in {elapsed:.0f}s")
    print(f"  Memory: {len(memory)} patterns, importance: {memory.importance_stats()}")
    env.close()


# ═══════════════════════════════════════════════════════════════════
# SLEEP CONSOLIDATION (R4 — non-negotiable)
# ═══════════════════════════════════════════════════════════════════
#
# Maps to brain (hippocampal sharp-wave ripple replay + STC):
#   1. Sample (state, action, Q) from memory proportional to importance²
#   2. Train slow system (value network) to predict Q from state
#   3. The slow system gradually absorbs knowledge from fast memory
#   4. Recompute position keys after consolidation

class SlowValueNet(nn.Module):
    """Slow-learning value network (neocortex analogue).
    Learns to predict Q-values from state vectors via replay.
    """
    def __init__(self, state_dim=12, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sleep_consolidation(memory, sf_net, slow_opt, dg_pos, cfg):
    """Replay stored φ(s) to train the successor feature network.
    The slow system learns the predictive map (environment dynamics).
    """
    losses = []
    for _ in range(100):
        keys, states, phis = memory.sample_prioritized(
            cfg.minibatch_size, temperature=0.5)
        if keys is None:
            break

        # Predict φ(s) from state — trains the predictive map
        phi_pred = sf_net(states.to(DEVICE))
        phi_target = phis.to(DEVICE)
        loss = F.mse_loss(phi_pred, phi_target)

        slow_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(slow_net.parameters(), 1.0)
        slow_opt.step()
        losses.append(loss.item())

    # Recompute keys from stored positions (ensures consistency)
    memory.recompute_keys(dg_pos)

    if losses:
        return np.mean(losses)
    return 0.0


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main()
