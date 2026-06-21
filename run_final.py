"""Final: PPO + GRU + memory retrieval + OR-gate store.

Key fix: GRU states are properly managed by storing hidden_before
each step and re-running the GRU in minibatches during PPO update.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import gymnasium as gym
import gymnasium_robotics
from PIL import Image
from pathlib import Path

OUT = Path('results/final')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MazeAutoencoder(nn.Module):
    """Frozen encoder — loads pretrained weights."""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(),
            nn.Linear(256, 2048), nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2, output_padding=1), nn.Sigmoid(),
        )
        self.encoder_bn = nn.BatchNorm1d(feature_dim)
        self.load()
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def load(self):
        self.load_state_dict(torch.load(
            'results/step2_maze/autoencoder_maze.pt',
            map_location=DEVICE, weights_only=True), strict=False)

    def encode(self, x):
        return self.encoder_bn(self.encoder(x))

    def decode(self, h):
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x)), self.encode(x)


class PolicyGRU(nn.Module):
    """GRU policy: input_dim → GRU(128) → mean(2) + value(1)."""

    def __init__(self, input_dim=256, hidden_dim=128, action_dim=2):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, x, h):
        h = self.gru(x, h)
        mean = torch.tanh(self.mean(h))
        std = F.softplus(self.log_std) + 1e-4
        val = self.value(h).squeeze(-1)
        return mean, std, val, h

    def reset(self, b=1):
        return torch.zeros(b, self.gru.hidden_size, device=DEVICE)


class EpisodicMemory:
    """Memory with OR-gate storage and retrieval."""

    def __init__(self, max_size=500):
        self.keys = []; self.vals = []; self.obs = []
        self.fbuf = []; self.max_size = max_size

    def store(self, k, v, o):
        if len(self.keys) >= self.max_size:
            self.keys.pop(0); self.vals.pop(0); self.obs.pop(0)
        self.keys.append(k.cpu()); self.vals.append(v.cpu()); self.obs.append(o.cpu())

    def sample(self, n, dev='cpu'):
        if not self.obs: return None, None, None
        idx = np.random.choice(len(self.obs), min(n, len(self.obs)), replace=False)
        return (torch.stack([self.obs[i] for i in idx]).to(dev),
                torch.stack([self.keys[i] for i in idx]).to(dev),
                torch.stack([self.vals[i] for i in idx]).to(dev))

    def retrieve(self, zq):
        if not self.keys: return None
        Z = torch.stack(self.keys, dim=0).to(zq.device)
        V = torch.stack(self.vals, dim=0).to(zq.device)
        ov = (zq * Z).sum(dim=-1)  # (N,) overlaps, one per stored pattern
        return V[ov.argmax()].unsqueeze(0)

    def recompute(self, enc, sep):
        if not self.obs: return
        dev = enc.encoder[0].weight.device; enc.eval()
        with torch.no_grad():
            all_h = enc.encode(torch.stack(self.obs).to(dev))
            for i in range(len(self.obs)):
                z = sep(all_h[i].unsqueeze(0)).squeeze(0)
                self.keys[i] = z.cpu(); self.vals[i] = all_h[i].cpu()

    def fast_nov(self, h):
        self.fbuf.append(h.cpu())
        if len(self.fbuf) <= 10: return 1.0
        c = F.cosine_similarity(h.unsqueeze(0), self.fbuf[-10].unsqueeze(0).to(h.device))
        return max(0.0, 1.0 - c.item())

    def __len__(self): return len(self.keys)


def make_env():
    return gym.make('PointMaze_UMazeDense-v3', render_mode='rgb_array')


def get_frame(env):
    raw = env.render()
    img = Image.fromarray(raw).resize((64, 64))
    return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).float().to(DEVICE)


def compute_gae(rew, val, done, g=0.99, l=0.95):
    adv = torch.zeros_like(rew)
    last = 0
    for t in reversed(range(len(rew))):
        delta = rew[t] + g * val[t + 1] * (1 - done[t]) - val[t]
        last = delta + g * l * (1 - done[t]) * last
        adv[t] = last
    return adv, adv + val[:-1]


def run(label, use_cls=True, use_gru=True, n_steps=100000):
    print(f"\n===== {label} =====")
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)

    ae = MazeAutoencoder().to(DEVICE)
    from hopfield_memory import PatternSeparator
    dg = PatternSeparator(128, 2000, 0.02)

    inp_dim = 256 if (use_cls and use_gru) else 128
    policy = PolicyGRU(input_dim=inp_dim).to(DEVICE) if use_gru else None
    if not use_gru:
        from step3_mujoco_ppo import PolicyHead
        policy = PolicyHead(hidden_dim=inp_dim).to(DEVICE)

    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    mem = EpisodicMemory(max_size=500) if use_cls else None

    env = make_env()
    env.reset()
    frame = get_frame(env)
    h = policy.reset(1) if use_gru else None
    traj = []
    ep = 0
    fh = deque(maxlen=500)
    sh = deque(maxlen=500)
    sc = 0
    global_step = 0

    while global_step < n_steps:
        # ── Wake ──────────────────────────────────────────────────────
        data = {k: [] for k in ['pi', 'h0', 'a', 'logp', 'v', 'r', 'd', 'er']}

        for _ in range(min(1024, n_steps - global_step)):
            with torch.no_grad():
                recon, feat = ae(frame.unsqueeze(0))
                if use_cls:
                    z = dg(feat)
                    hr = mem.retrieve(z)
                    if hr is not None:
                        pi = torch.cat([feat.squeeze(0), hr.squeeze(0)]).unsqueeze(0)
                    else:
                        pi = torch.cat([feat.squeeze(0), torch.zeros(128, device=DEVICE)]).unsqueeze(0)
                else:
                    pi = feat
                h0 = h.clone() if use_gru else None
                if use_gru:
                    m, s, v, h = policy(pi, h)
                else:
                    m = torch.tanh(policy(pi))
                    s = F.softplus(policy.log_std) + 1e-4
                    v = policy.value(pi).squeeze(-1)
                d = Normal(m, s)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)

            obs, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            done = term or trunc
            nf = get_frame(env)

            if use_cls:
                fn = mem.fast_nov(feat.squeeze(0))
                sn = F.mse_loss(recon, frame.unsqueeze(0)).item()
                fh.append(fn); sh.append(sn)
                r_total = re + 0.01 * (0.5 * fn + 0.5 * sn)
                if len(fh) > 20:
                    if (fn > np.percentile(fh, 75)) or (sn > np.percentile(sh, 75)):
                        mem.store(z.squeeze(0), feat.squeeze(0), frame.cpu())
                else:
                    mem.store(z.squeeze(0), feat.squeeze(0), frame.cpu())
            else:
                r_total = re

            data['pi'].append(pi.squeeze(0))
            if use_gru: data['h0'].append(h0.squeeze(0))
            data['a'].append(a.squeeze(0))
            data['logp'].append(lp)
            data['v'].append(v)
            data['r'].append(r_total)
            data['d'].append(1 if done else 0)
            data['er'].append(re)

            frame = nf
            traj.append(env.unwrapped.sim.data.qpos[:2].copy() if hasattr(env.unwrapped, 'sim') else [0, 0])
            global_step += 1

            if done:
                env.reset()
                frame = get_frame(env)
                h = policy.reset(1) if use_gru else None
                ep += 1

        # ── PPO ──────────────────────────────────────────────────────
        pi_s = torch.stack(data['pi'])
        a_s = torch.stack(data['a'])
        lp_s = torch.stack(data['logp'])
        v_s = torch.stack(data['v'])
        d_s = torch.tensor(data['d'], device=DEVICE, dtype=torch.float32)
        r_s = torch.tensor(data['r'], device=DEVICE, dtype=torch.float32)
        r_n = (r_s - r_s.mean()) / (r_s.std() + 1e-8)

        # Bootstrap
        with torch.no_grad():
            if use_cls:
                z2 = dg(ae.encode(frame.unsqueeze(0)))
                hr2 = mem.retrieve(z2)
                pi2 = torch.cat([ae.encode(frame.unsqueeze(0)).squeeze(0), hr2.squeeze(0) if hr2 is not None else torch.zeros(128, device=DEVICE)]).unsqueeze(0)
            else:
                pi2 = ae.encode(frame.unsqueeze(0))
            if use_gru:
                _, _, nv, _ = policy(pi2, h)
            else:
                nv = policy.value(pi2).squeeze(-1)

        av = torch.cat([v_s.view(-1), nv.view(-1)])
        adv, ret = compute_gae(r_n, av, d_s)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO minibatch
        for _ in range(4):
            perm = torch.randperm(len(pi_s))
            for i in range(0, len(pi_s), 64):
                idx = perm[i:i+64]
                pi_m = pi_s[idx]

                if use_gru:
                    h_m = policy.reset(len(idx))
                    # Run GRU for each step independently from reset
                    # (approximates temporal context from stored h0)
                    # Proper: use stored h0[idx] as initial state
                    if 'h0' in data and len(data['h0']) > 0:
                        h0_s = torch.stack(data['h0'])
                        h_m = h0_s[idx]
                    m2, s2, v2, _ = policy(pi_m, h_m)
                else:
                    m2 = torch.tanh(policy(pi_m))
                    s2 = F.softplus(policy.log_std) + 1e-4
                    v2 = policy.value(pi_m).squeeze(-1)

                d2 = Normal(m2, s2)
                lp2 = d2.log_prob(a_s[idx]).sum(-1)
                ent = d2.entropy().sum(-1).mean()

                ratio = (lp2 - lp_s[idx]).exp()
                ca = torch.clamp(ratio, 0.8, 1.2) * adv[idx]
                pl = -(torch.min(ratio * adv[idx], ca)).mean()
                vl = F.mse_loss(v2, ret[idx])
                loss = pl + 0.5 * vl - 0.001 * ent  # Reduced entropy coefficient

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()

        # ── Sleep ─────────────────────────────────────────────────────
        if use_cls:
            sc += 1
            if sc % 4 == 0 and len(mem) >= 32:
                for p in ae.parameters(): p.requires_grad_(True)
                ae.train()
                eo = torch.optim.Adam(ae.parameters(), lr=1e-5)
                for _ in range(100):
                    om, _, _ = mem.sample(64, DEVICE)
                    if om is None: break
                    eo.zero_grad()
                    F.mse_loss(ae(om)[0], om).backward()
                    eo.step()
                ae.eval()
                for p in ae.parameters(): p.requires_grad_(False)
                mem.recompute(ae, dg)

        # ── Log ──────────────────────────────────────────────────────
        if global_step % 2048 == 0:
            print(f"  {label:>20} step={global_step:6d} rew={np.mean(data['er']):.4f} "
                  f"mem={len(mem) if use_cls else 0} ep={ep}")

        if global_step % 20000 == 0 or global_step >= n_steps - 1024:
            plot_heat(traj, global_step, label, out)

    env.close()
    plot_heat(traj, n_steps, label, out)
    torch.save(policy.state_dict(), out / 'policy.pt')
    print(f"  {label} done. {global_step} steps, {ep} episodes.")


def plot_heat(traj, step, label, p):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array(traj)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(t[:, 0], t[:, 1], gridsize=20, cmap='Blues', mincnt=1, alpha=0.8)
    ax.scatter(t[0, 0], t[0, 1], c='green', s=80, marker='*', zorder=5)
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5); ax.set_aspect('equal')
    ax.set_title(f'{label} step {step}')
    plt.tight_layout(); plt.savefig(p / f'traj_{step:06d}.png', dpi=100); plt.close()


if __name__ == '__main__':
    run('PPO_GRU_baseline', use_cls=False, use_gru=True, n_steps=100000)
    run('Percepta_full', use_cls=True, use_gru=True, n_steps=100000)
