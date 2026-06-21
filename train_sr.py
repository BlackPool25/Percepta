"""Successor Representation training: allocentric predictive map + adjustable goal.
Q(s) = φ(s)^T · w
φ(s) = successor features (predictive map) — stable across goal changes
w = reward weights — the ONLY thing that changes when goal moves

Curriculum: fixed start → random start → random goal → random maze
"""

import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from collections import deque

from hopfield_memory import PatternSeparator
from env_nav import NavArena

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('results/sr')
OUT.mkdir(parents=True, exist_ok=True)
STATE_DIM = 12
PHI_DIM = 64
HIDDEN = 64

# ─── Successor Feature Network ────────────────────────────────────
class SRNet(nn.Module):
    """Q(s) = φ(s)^T · w. φ(s) is the predictive map, w is the goal."""
    def __init__(self):
        super().__init__()
        self.phi_net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, PHI_DIM),
        )
        self.w = nn.Parameter(torch.zeros(PHI_DIM))

    def phi(self, s):
        return self.phi_net(s)

    def q(self, phi):
        return (phi * self.w.unsqueeze(0)).sum(dim=-1)

    def q_from_state(self, s):
        return self.q(self.phi(s))

# ─── Policy ────────────────────────────────────────────────────────
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM + 2, HIDDEN), nn.ReLU())
        self.mean = nn.Linear(HIDDEN, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.value = nn.Linear(HIDDEN, 1)
        nn.init.orthogonal_(self.mean.weight, 0.01)
        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, x):
        h = self.net(x)
        m = torch.tanh(self.mean(h))
        s = F.softplus(self.log_std) + 1e-4
        v = self.value(h).squeeze(-1)
        return m, s, v

# ─── Memory (stores φ(s) — allocentric predictive map) ────────────
class SRMemory:
    def __init__(self, max_size=1000, beta=2.0):
        self.keys, self.states, self.phis = [], [], []
        self.importance = []
        self.max_size, self.beta = max_size, beta
        self.last_retrieved = None

    def store(self, key, state, phi):
        if len(self.keys) >= self.max_size:
            idx = int(np.argmin(self.importance))
            self.keys.pop(idx); self.states.pop(idx)
            self.phis.pop(idx); self.importance.pop(idx)
        self.keys.append(key.cpu().detach())
        self.states.append(state.cpu().detach())
        self.phis.append(phi.cpu().detach())
        self.importance.append(0.5)

    def retrieve(self, q):
        if not self.keys:
            self.last_retrieved = None
            return torch.zeros(1, STATE_DIM, device=q.device), \
                   torch.zeros(1, PHI_DIM, device=q.device), 0.0
        Z = torch.stack(self.keys).to(q.device)
        S = torch.stack(self.states).to(q.device)
        P = torch.stack(self.phis).to(q.device)
        logits = self.beta * (q @ Z.T)
        attn = F.softmax(logits, dim=-1)
        # RBF similarity (not diluted by pattern count)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        Zn = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)
        rbf = torch.exp(-2.0 * (1.0 - (qn @ Zn.T).squeeze(0))).max().item()
        # Track best matching index for TD update
        self.last_retrieved = attn.argmax().item() if attn.max().item() > 0.1 else None
        self._update_imp(attn.squeeze(0))
        return attn @ S, attn @ P, rbf

    def _update_imp(self, w):
        ws = w.detach().cpu().numpy()
        for i, v in enumerate(ws):
            if v > 0.05:
                self.importance[i] = min(self.importance[i] + 0.05, 1.0)

    def decay_imp(self, f=0.99):
        for i in range(len(self.importance)):
            self.importance[i] = max(self.importance[i] * f, 0.01)

    def sample(self, n, t=0.5):
        if not self.states: return None, None, None
        imp = np.clip(np.array(self.importance, np.float64), 0.01, 1.0)
        p = imp ** (1/t)
        p /= p.sum() + 1e-10
        idx = np.random.choice(len(self.states), min(n, len(self.states)), False, p)
        return (torch.stack([self.keys[i] for i in idx]),
                torch.stack([self.states[i] for i in idx]),
                torch.stack([self.phis[i] for i in idx]))

    def __len__(self): return len(self.keys)


def compute_gae(r, v, d, g=0.99, l=0.95):
    adv = torch.zeros_like(r)
    last = 0.0
    for t in reversed(range(len(r))):
        δ = r[t] + g * v[t+1] * (1 - d[t]) - v[t]
        last = δ + g * l * (1 - d[t]) * last
        adv[t] = last
    return adv, adv + v[:-1]


def train(phases, steps_per_phase):
    """Run full SR curriculum training."""
    dg = PatternSeparator(2, 2000, 0.02)
    mem = SRMemory(max_size=2000)
    sf = SRNet().to(DEVICE)
    pi = Policy().to(DEVICE)
    opt = torch.optim.Adam(list(pi.parameters()) + list(sf.parameters()), lr=3e-4)
    w_opt = torch.optim.Adam([sf.w], lr=1e-2)  # faster LR for w (goal adaptation)
    env = NavArena(render_mode='rgb_array')

    for pid, n_steps in zip(phases, steps_per_phase):
        env.set_curriculum(pid)
        print(f"\n=== PHASE {pid} ({n_steps} steps) ===")
        obs, _ = env.reset()
        s = obs['state']
        roll = {k: [] for k in ['s','gd','a','lp','v','r','d','zk','ca1','qt']}
        goals, eps = 0, 0
        step = 0

        while step < n_steps:
            with torch.no_grad():
                st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
                pos, g = st[:, :2], st[:, 2:4]
                gd = (g - pos) / ((g - pos).norm(dim=-1, keepdim=True) + 1e-8)
                z = dg(pos)
                ss, ps, rbf = mem.retrieve(z)

                # Q(s) = φ(s)^T · w — computed from successor features
                phi_s = sf.phi(st)
                q_episodic = sf.q(ps).item()
                q_parametric = sf.q(phi_s).item()
                blend = rbf if rbf > 0.05 else 0.0
                q_target = blend * q_episodic + (1 - blend) * q_parametric

                ca1 = (pos - ss[:, :2]).norm().item()
                pi_in = torch.cat([st.squeeze(0), gd.squeeze(0)]).unsqueeze(0)
                m, sd, v = pi(pi_in)
                d2 = Normal(m, sd)
                a = d2.sample()
                lp = d2.log_prob(a).sum(-1)

            obs2, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            done = term or trunc
            s2 = obs2['state']
            nxt = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

            # TD on w: w ← w + α·(r + γ·Q(s') - Q(s))·φ(s)
            with torch.no_grad():
                phi_next = sf.phi(nxt)
                q_next = sf.q(phi_next).item()
            td_target = re + 0.99 * q_next
            td_error = td_target - q_parametric
            sf.w.data += 1e-2 * td_error * phi_s.squeeze(0).detach()

            # Successor loss: φ(s) should predict φ(s) + γ·φ(s')
            phi_target = phi_s + 0.99 * phi_next.detach()
            sf_loss = F.mse_loss(phi_s, phi_target.detach())

            # Store φ(s) in memory
            mem.store(z.squeeze(0), st.squeeze(0), phi_s.squeeze(0))
            mem.decay_imp()

            total_reward = re + max(0.05, 0.3 * (1 - step / n_steps)) * ca1
            roll['s'].append(st.squeeze(0))
            roll['gd'].append(gd.squeeze(0))
            roll['a'].append(a.squeeze(0))
            roll['lp'].append(lp)
            roll['v'].append(v)
            roll['r'].append(total_reward)
            roll['d'].append(1.0 if done else 0.0)
            roll['zk'].append(z.squeeze(0))
            roll['ca1'].append(ca1)
            roll['qt'].append(q_target)

            s = s2
            step += 1
            if done:
                eps += 1
                if term: goals += 1
                if eps % 10 == 0:
                    print(f"  step {step}: ext={np.mean(roll['r'][-512:]):.2f}, "
                          f"Qtarget={np.mean(roll['qt'][-512:]):.1f}, "
                          f"goals={goals}/{eps}")
                    if goals >= 5: break  # advance if mastering
                obs, _ = env.reset(); s = obs['state']

        # PPO update
        if len(roll['s']) >= 64:
            sb = torch.stack(roll['s'])
            gb = torch.stack(roll['gd'])
            ab = torch.stack(roll['a'])
            ob = torch.stack(roll['lp'])
            rb = torch.tensor(roll['r'], device=DEVICE, dtype=torch.float32)
            db = torch.tensor(roll['d'], device=DEVICE, dtype=torch.float32)
            vb = torch.stack(roll['v'])
            with torch.no_grad():
                _, _, nv = pi(torch.cat([sb[-1:], gb[-1:]], dim=-1))
            av = torch.cat([vb.view(-1), nv.view(-1)])
            adv, ret = compute_gae(rb, av, db)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(4):
                perm = torch.randperm(len(sb))
                for i in range(0, len(sb), 64):
                    idx = perm[i:i+64]
                    m, sd, vm = pi(torch.cat([sb[idx], gb[idx]], dim=-1))
                    d2 = Normal(m, sd)
                    lp2 = d2.log_prob(ab[idx]).sum(-1)
                    ratio = (lp2 - ob[idx]).exp()
                    ca2 = torch.clamp(ratio, 0.8, 1.2) * adv[idx]
                    pl = -(torch.min(ratio * adv[idx], ca2)).mean()
                    vl = F.mse_loss(vm, ret[idx])
                    # Combine PPO loss + SR loss
                    phi_b = sf.phi(sb[idx])
                    phi_b_next = sf.phi(sb[idx].roll(-1, dims=0))
                    sr_loss = F.mse_loss(phi_b, (phi_b + 0.99 * phi_b_next.detach()))
                    loss = pl + 0.5 * vl - 0.01 * d2.entropy().sum(-1).mean() + 0.01 * sr_loss
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
                    opt.step()

        # Sleep consolidation (replay stored φ to stabilize φ_net)
        for _ in range(50):
            ks, ss, phis = mem.sample(64)
            if ks is None: break
            pred = sf.phi(ss.to(DEVICE))
            l = F.mse_loss(pred, phis.to(DEVICE))
            opt.zero_grad(); l.backward(); opt.step()

        print(f"  Phase {pid} done: {goals}/{eps} goal rate={goals/max(eps,1):.0%}")

    env.close()
    torch.save(pi.state_dict(), OUT / 'policy.pt')
    torch.save(sf.state_dict(), OUT / 'sr_net.pt')
    print(f"\nSaved to {OUT}")


def test(phases, n=500):
    """Test on multiple phases."""
    dg = PatternSeparator(2, 2000, 0.02)
    mem = SRMemory(max_size=2000)
    sf = SRNet().to(DEVICE); sf.load_state_dict(torch.load(OUT / 'sr_net.pt', map_location=DEVICE))
    pi = Policy().to(DEVICE); pi.load_state_dict(torch.load(OUT / 'policy.pt', map_location=DEVICE))
    env = NavArena(render_mode='rgb_array')
    print(f"\n{'='*50}\nTESTING\n{'='*50}")
    for pid in phases:
        env.set_curriculum(pid)
        goals, eps = 0, 0
        for ep in range(n):
            obs, _ = env.reset(); s = obs['state']
            reached = False
            for _ in range(500):
                with torch.no_grad():
                    st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
                    gd = (st[:,2:4] - st[:,:2]) / ((st[:,2:4] - st[:,:2]).norm(dim=-1, keepdim=True) + 1e-8)
                    pi_in = torch.cat([st.squeeze(0), gd.squeeze(0)]).unsqueeze(0)
                    m, sd, _ = pi(pi_in)
                    a = Normal(m, sd).sample()
                obs2, _, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
                s = obs2['state']
                if term: reached = True; break
                if trunc: break
            if reached: goals += 1
            eps += 1
        print(f"  Phase {pid}: goal rate={goals/eps:.0%} ({goals}/{eps})")
    env.close()


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    train(phases=[0, 1, 2, 3], steps_per_phase=[3000, 5000, 8000, 10000])
    test(phases=[0, 1, 2, 3], n=500)
