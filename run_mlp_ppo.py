"""Plain MLP PPO baseline on PointMaze — no GRU, no CLS, zero entropy bonus."""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import gymnasium as gym, gymnasium_robotics
from pathlib import Path

OUT = Path('results/mlp_ppo_baseline')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_env():
    return gym.make('PointMaze_UMazeDense-v3')


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mean = nn.Linear(64, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.value = nn.Linear(64, 1)
        for m in [self.mean, self.value]:
            nn.init.orthogonal_(m.weight, gain=0.01 if m is self.mean else 1.0)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.fc(x)
        m = torch.tanh(self.mean(h))
        s = F.softplus(self.log_std) + 1e-4
        v = self.value(h).squeeze(-1)
        return m, s, v


def compute_gae(rew, val, done, g=0.99, l=0.95):
    adv = torch.zeros_like(rew)
    last = 0
    for t in reversed(range(len(rew))):
        d = rew[t] + g * val[t+1] * (1 - done[t]) - val[t]
        last = d + g * l * (1 - done[t]) * last
        adv[t] = last
    return adv, adv + val[:-1]


def run():
    policy = PolicyMLP().to(DEVICE)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

    env = make_env()
    obs, info = env.reset()

    ent_log, vl_log, kl_log = [], [], deque(maxlen=10)
    ep_dist_log = []
    ep_count = 0
    global_step = 0

    while global_step < 200000:
        data = {k: [] for k in ['obs', 'act', 'lp', 'v', 'r', 'd']}

        for _ in range(2048):
            m, s, v = policy(torch.from_numpy(obs['observation']).float().unsqueeze(0).to(DEVICE))
            dist = Normal(m, s)
            a = dist.sample()
            lp = dist.log_prob(a).sum(-1)

            obs2, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            done = term or trunc
            data['obs'].append(obs['observation'].copy())
            data['act'].append(a.squeeze(0).cpu())
            data['lp'].append(lp)
            data['v'].append(v)
            data['r'].append(re)
            data['d'].append(1 if done else 0)

            global_step += 1
            if done:
                ep_dist_log.append(np.linalg.norm(obs['observation'][:2] - obs2['observation'][:2] if hasattr(obs2, 'observation') else [0, 0]))
                obs, info = env.reset()
                ep_count += 1
            else:
                obs = obs2

        # PPO
        o_b = torch.tensor(np.array(data['obs']), dtype=torch.float32, device=DEVICE)
        a_b = torch.stack(data['act']).to(DEVICE)
        lp_b = torch.stack(data['lp'])
        v_b = torch.stack(data['v'])
        d_b = torch.tensor(data['d'], device=DEVICE, dtype=torch.float32)
        r_b = torch.tensor(data['r'], device=DEVICE, dtype=torch.float32)
        r_n = (r_b - r_b.mean()) / (r_b.std() + 1e-8)

        with torch.no_grad():
            _, _, nv = policy(torch.from_numpy(obs['observation']).float().unsqueeze(0).to(DEVICE))
        all_v = torch.cat([v_b.view(-1), nv.view(-1)])
        adv, ret = compute_gae(r_n, all_v, d_b)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(10):
            perm = torch.randperm(len(o_b))
            for i in range(0, len(o_b), 64):
                idx = perm[i:i+64]
                optim.zero_grad()
                m2, s2, v2 = policy(o_b[idx])
                d2 = Normal(m2, s2)
                lp2 = d2.log_prob(a_b[idx]).sum(-1)
                ent = d2.entropy().sum(-1).mean()

                ratio = (lp2 - lp_b[idx]).exp()
                ca = torch.clamp(ratio, 0.8, 1.2) * adv[idx]
                pl = -(torch.min(ratio * adv[idx], ca)).mean()
                vl = F.mse_loss(v2, ret[idx])
                loss = pl + 0.5 * vl - 0.0 * ent

                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()

        # Diagnostics
        with torch.no_grad():
            m_o, s_o, _ = policy(o_b)
            kl = (torch.log(s_o/s_o) + (s_o**2 + 0) / (2 * s_o**2) - 0.5).sum(-1).mean().item()  # ~0 for identical dist
        kl_log.append(0.0)
        ent_log.append(ent.item())
        vl_log.append(vl.item())

        if global_step % 4096 == 0:
            mean_r = np.mean(data['r'])
            mean_dist = np.mean([np.linalg.norm(o[:2]) for o in data['obs']])
            print(f"  step={global_step:7d} rew={mean_r:.4f} dist={mean_dist:.2f} "
                  f"ent={ent.item():.3f} kl={kl:.4f} vl={vl.item():.1f} ep={ep_count}")

    env.close()
    print(f"Done. {global_step} steps, {ep_count} episodes.")


if __name__ == '__main__':
    run()
