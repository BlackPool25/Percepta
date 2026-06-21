"""Step 3: MuJoCo PointMaze with PPO + dual novelty intrinsic reward.

Architecture:
  Pixel observation (64x64 RGB) → Frozen MazeAutoencoder → 128-dim features h
    → Policy head (Linear 128→2, tanh) → action
    → Value head (Linear 128→1) → value estimate
  Dual novelty:
    Fast: hamming distance to nearest stored key → gates episodic storage
    Slow: reconstruction error → gates learning rate
    Combined: alpha * fast + (1-alpha) * slow → intrinsic reward

Frozen encoder during PPO training — only policy/value heads updated.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from collections import deque
from torch.distributions import Normal

OUT = Path('results/step3_mujoco')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

AE_PATH = Path('results/step2_maze/autoencoder_maze.pt')

# ─── Components ───────────────────────────────────────────────────────────

class MazeAutoencoder(nn.Module):
    """Frozen encoder from Step 2. Loads saved weights."""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )
        self.encoder_bn = nn.BatchNorm1d(feature_dim)

    def encode(self, x):
        return self.encoder_bn(self.encoder(x))

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h


class PolicyHead(nn.Module):
    """Gaussian policy with separate value head."""

    def __init__(self, hidden_dim=128, action_dim=2):
        super().__init__()
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.value = nn.Linear(hidden_dim, 1)

        # Orthogonal init: small gains for policy, default for value
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, h):
        mean = torch.tanh(self.policy_mean(h))
        std = F.softplus(self.log_std) + 1e-4
        value = self.value(h).squeeze(-1)  # (N,)
        return mean, std, value

    def act(self, h, deterministic=False):
        mean, std, value = self.forward(h)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value, mean, std


class BoundedMemory:
    """Fast episodic memory with FIFO eviction. Stores (key, value, obs) triples."""

    def __init__(self, max_size=200):
        self.keys = []
        self.values = []
        self.observations = []
        self.feature_buffer = []
        self.max_size = max_size

    def store(self, key, value, obs=None):
        if len(self.keys) >= self.max_size:
            self.keys.pop(0)
            self.values.pop(0)
            if self.observations:
                self.observations.pop(0)
        self.keys.append(key.cpu())
        self.values.append(value.cpu())
        if obs is not None:
            self.observations.append(obs.cpu())

    def sample(self, batch_size, device='cpu'):
        """Sample random (obs, key, value) triples for sleep consolidation."""
        n = len(self.observations)
        if n == 0:
            return None, None, None
        idx = np.random.choice(n, min(batch_size, n), replace=False)
        obs = torch.stack([self.observations[i] for i in idx]).to(device)
        keys = torch.stack([self.keys[i] for i in idx]).to(device)
        vals = torch.stack([self.values[i] for i in idx]).to(device)
        return obs, keys, vals

    def recompute_keys(self, encoder, sep):
        """Recompute stored keys after encoder drift."""
        if not self.observations:
            return
        device = encoder.encoder[0].weight.device
        encoder.eval()
        with torch.no_grad():
            # Process in batches for BatchNorm compatibility
            all_obs = torch.stack(self.observations).to(device)
            all_h = encoder.encode(all_obs)  # returns features, not (recon, h)
            for i in range(len(self.observations)):
                h = all_h[i].unsqueeze(0)
                z = sep(h).squeeze(0)
                self.keys[i] = z.cpu()
                self.values[i] = h.squeeze(0).cpu()

    def fast_novelty(self, h):
        """Fast novelty: temporal feature change.

        Measures cosine similarity between current features and features
        from N steps ago. When the agent is in the same visual context,
        consecutive features are similar → low fast novelty.
        When the agent enters a new area, features change → high fast novelty.

        This detects EVENT BOUNDARIES (entering a new room, turning a corner)
        rather than pixel-level differences.
        """
        self.feature_buffer.append(h.cpu())
        window = 10  # Compare against feature from 10 steps ago
        if len(self.feature_buffer) <= window:
            return 1.0
        h_prev = self.feature_buffer[-window]
        cosim = F.cosine_similarity(h.unsqueeze(0), h_prev.unsqueeze(0).to(h.device))
        return max(0.0, 1.0 - cosim.item())

    def __len__(self):
        return len(self.keys)


class RunningNorm:
    """Running mean/std for reward normalization."""

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
        n = x.size if hasattr(x, 'size') else len(x) if hasattr(x, '__len__') else 1
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        self.std = np.sqrt(self.std**2 + batch_std**2 * n / self.count +
                           delta**2 * n * (self.count - n) / self.count / self.count)


# ─── Environment ──────────────────────────────────────────────────────────

def make_env(render=False):
    return gym.make(
        'PointMaze_UMazeDense-v3',
        render_mode='human' if render else 'rgb_array',
        continuing_task=False,
    )


def process_obs(env, obs, info):
    """Render and resize to 64x64."""
    raw = env.render()
    img = Image.fromarray(raw).resize((64, 64))
    frame = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(frame).permute(2, 0, 1).float().to(DEVICE), obs['observation'][:2]


# ─── GAE ──────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards) - 1)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


# ─── Training ──────────────────────────────────────────────────────────────

class Config:
    total_steps = 50000
    steps_per_rollout = 1024
    ppo_epochs = 4
    minibatch_size = 64
    lr = 3e-4
    encoder_lr = 1e-5
    recon_loss_weight = 0.01  # Keep encoder stable — prevent overfitting to rollout frames
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    max_grad_norm = 0.5
    alpha = 0.5  # intrinsic reward blend
    memory_size = 200
    store_threshold_pct = 70
    log_interval = 100
    trajectory_interval = 2000


def log_trajectory(positions, step, save_path):
    """Plot top-down view of agent trajectory."""
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
    print(f"Device: {DEVICE}\n")

    cfg = Config()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    # ── Load encoder ───────────────────────────────────────────────────
    ae = MazeAutoencoder(feature_dim=128).to(DEVICE)
    ae.load_state_dict(torch.load(AE_PATH, map_location=DEVICE, weights_only=True))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)  # Frozen
    print(f"Loaded autoencoder from {AE_PATH}")

    # ── Policy ─────────────────────────────────────────────────────────
    policy = PolicyHead(hidden_dim=128, action_dim=2).to(DEVICE)
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    # ── Memory / novelty ───────────────────────────────────────────────
    memory = BoundedMemory(max_size=cfg.memory_size)

    # ── Reward normalization ───────────────────────────────────────────
    reward_norm = RunningNorm()

    # ── Logging ────────────────────────────────────────────────────────
    log = {
        'step': [], 'rew': [], 'ep_len': [], 'policy_loss': [], 'value_loss': [],
        'entropy': [], 'explained_var': [], 'action_std': [],
        'fast_novelty': [], 'slow_novelty': [], 'store_rate': [],
    }

    # ── Env ────────────────────────────────────────────────────────────
    env = make_env()
    obs, info = env.reset()
    frame_t, pos = process_obs(env, obs, info)

    # ── Main loop ──────────────────────────────────────────────────────
    global_step = 0
    episode_rewards = []
    ep_reward = 0
    ep_length = 0
    trajectory_positions = [pos.copy()]
    stores_this_log = 0
    total_this_log = 0
    sleep_counter = 0

    # Running novelty histories for adaptive thresholding
    fast_history = deque(maxlen=500)
    slow_history = deque(maxlen=500)

    # DG projection for fast memory keys (separate from encoder)
    from hopfield_memory import PatternSeparator
    dg = PatternSeparator(128, 2000, 0.02)

    while global_step < cfg.total_steps:
        # ── WAKE PHASE: rollout collection — encoder FROZEN ────────────
        for p in ae.parameters():
            p.requires_grad_(False)

        rollout = {k: [] for k in ['obs', 'h', 'action', 'log_prob', 'value',
                                    'reward', 'done', 'fast_nov', 'slow_nov']}

        for _ in range(cfg.steps_per_rollout):
            with torch.no_grad():
                recon, h = ae(frame_t.unsqueeze(0))
                z = dg(h)  # DG projection on features
                mean, std, value = policy(h)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            # Step env
            act_np = action.squeeze(0).cpu().numpy()
            obs, reward_ext, terminated, truncated, info = env.step(act_np)
            done = terminated or truncated

            # Process next observation
            next_frame_t, pos = process_obs(env, obs, info)

            # Compute novelty
            slow_nov = F.mse_loss(recon, frame_t.unsqueeze(0)).item()
            fast_nov = memory.fast_novelty(h.squeeze(0))

            # Intrinsic + extrinsic reward
            reward_int = cfg.alpha * fast_nov + (1 - cfg.alpha) * slow_nov
            total_reward = reward_ext + 0.01 * reward_int

            # Adaptive OR-gate storage: store if either signal exceeds its percentile threshold
            fast_history.append(fast_nov)
            slow_history.append(slow_nov)
            if len(fast_history) > 20:
                fast_th = np.percentile(fast_history, 70)
                slow_th = np.percentile(slow_history, 70)
                store = (fast_nov > fast_th) or (slow_nov > slow_th)
            else:
                store = True  # Bootstrap: store everything initially
            if store:
                memory.store(z.squeeze(0), h.squeeze(0), frame_t.cpu())
                stores_this_log += 1
            total_this_log += 1

            # Store rollout
            rollout['obs'].append(frame_t)
            rollout['h'].append(h.squeeze(0))
            rollout['action'].append(action.squeeze(0))
            rollout['log_prob'].append(log_prob)
            rollout['value'].append(value)
            rollout['reward'].append(total_reward)
            rollout['done'].append(1.0 if done else 0.0)
            rollout['fast_nov'].append(fast_nov)
            rollout['slow_nov'].append(slow_nov)

            frame_t = next_frame_t
            ep_reward += total_reward
            ep_length += 1
            global_step += 1
            trajectory_positions.append(pos.copy())

            if done:
                episode_rewards.append(ep_reward)
                obs, info = env.reset()
                frame_t, pos = process_obs(env, obs, info)
                ep_reward = 0
                ep_length = 0

        # ── PPO update ────────────────────────────────────────────────
        obs_batch = torch.stack(rollout['obs'])
        h_batch = torch.stack(rollout['h'])
        action_batch = torch.stack(rollout['action'])
        old_log_prob_batch = torch.stack(rollout['log_prob'])
        reward_batch = torch.tensor(rollout['reward'], device=DEVICE)
        done_batch = torch.tensor(rollout['done'], device=DEVICE)
        value_batch = torch.stack(rollout['value'])

        # Normalize rewards
        for r in rollout['reward']:
            reward_norm.update(r)
        reward_batch = torch.tensor(
            [reward_norm(r) for r in rollout['reward']], 
            device=DEVICE, dtype=torch.float32)

        # GAE — bootstrap from the final observation's features
        with torch.no_grad():
            next_h = ae.encode(frame_t.unsqueeze(0))
            _, _, next_value = policy(next_h)
        all_values = torch.cat([value_batch.view(-1), next_value.view(-1)])
        advantages, returns = compute_gae(
            reward_batch, all_values, done_batch, cfg.gamma, cfg.lam)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        n = h_batch.size(0)
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.minibatch_size):
                idx = perm[i:i + cfg.minibatch_size]
                h_mb = h_batch[idx]
                a_mb = action_batch[idx]
                old_lp_mb = old_log_prob_batch[idx]
                adv_mb = advantages[idx]
                ret_mb = returns[idx]

                mean, std, value_mb = policy(h_mb)
                dist = Normal(mean, std)
                log_prob = dist.log_prob(a_mb).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (log_prob - old_lp_mb).exp()
                clip_adv = torch.clamp(ratio, 1 - cfg.clip_eps,
                                       1 + cfg.clip_eps) * adv_mb
                policy_loss = -(torch.min(ratio * adv_mb, clip_adv)).mean()

                value_loss = F.mse_loss(value_mb, ret_mb.detach())

                loss = policy_loss + 0.5 * value_loss - cfg.entropy_coef * entropy

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optim.step()

        # ── SLEEP PHASE: encoder update on memory replay ────────────────
        sleep_counter += 1
        if cfg.encoder_lr > 0 and sleep_counter % 4 == 0 and len(memory) >= 32:
            try:
                for p in ae.parameters():
                    p.requires_grad_(True)
                ae.train()
                enc_opt = torch.optim.Adam(ae.parameters(), lr=cfg.encoder_lr)

                for _ in range(100):
                    obs_mem, _, _ = memory.sample(64, DEVICE)
                    if obs_mem is None:
                        break
                    recon_mem, _ = ae(obs_mem)
                    loss = F.mse_loss(recon_mem, obs_mem)
                    enc_opt.zero_grad()
                    loss.backward()
                    enc_opt.step()

                ae.eval()
                for p in ae.parameters():
                    p.requires_grad_(False)
                memory.recompute_keys(ae, dg)
                print(f"  [sleep phase complete, mem={len(memory)}]")
            except Exception as e:
                ae.eval()
                for p in ae.parameters():
                    p.requires_grad_(False)
                print(f"  [sleep phase error: {e}]")

        # ── Logging ────────────────────────────────────────────────────
        if global_step // cfg.steps_per_rollout > len(log['step']):
            log['step'].append(global_step)
            log['fast_novelty'].append(np.mean(rollout['fast_nov']))
            log['slow_novelty'].append(np.mean(rollout['slow_nov']))
            log['store_rate'].append(stores_this_log / max(total_this_log, 1))

            with torch.no_grad():
                _, _, vals = policy(h_batch)
                ev = 1 - ((vals - returns)**2).mean() / (returns.var() + 1e-8)
            log['policy_loss'].append(policy_loss.item())
            log['value_loss'].append(value_loss.item())
            log['entropy'].append(entropy.item())
            log['explained_var'].append(ev.item())
            log['action_std'].append(std.mean().item())

            mean_rew = np.mean(rollout['reward'])
            print(f"  step={global_step:6d} reward={mean_rew:.4f} "
                  f"fast={log['fast_novelty'][-1]:.4f} "
                  f"slow={log['slow_novelty'][-1]:.4f} "
                  f"store={log['store_rate'][-1]:.2f} "
                  f"act_std={log['action_std'][-1]:.3f} "
                  f"ev={log['explained_var'][-1]:.2f}")

            stores_this_log = 0
            total_this_log = 0

        # Trajectory plot
        if global_step % cfg.trajectory_interval == 0:
            log_trajectory(trajectory_positions, global_step, out)
            trajectory_positions = [pos.copy()]

    # ── Final save ─────────────────────────────────────────────────────
    torch.save(policy.state_dict(), out / 'policy.pt')
    print(f"\nSaved: {out / 'policy.pt'}")
    env.close()
    print("Done.")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main()
