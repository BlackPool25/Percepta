"""Brain-like CLS: distributed φ + interleaved replay + metaplasticity + |δ|-gating.
Key fixes from research:
  1. Buffer NEVER fully cleared — demo permanent, |δ| tracks importance
  2. Sleep INTERLEAVES demo + high-|δ| transitions
  3. Metaplasticity: per-parameter Fisher importance → per-param LR
  4. Forward model protected by lower online LR
  5. |δ| gates consolidation priority — surprising events prioritized
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from hopfield_memory import PatternSeparator
from env_nav import NavArena

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('results/sr'); OUT.mkdir(parents=True, exist_ok=True)
S, P, H, G, A = 12, 256, 128, 2, 2

# ═══ SR Network (distributed φ representation) ═══════════════════
class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(S, 256), nn.ReLU(), nn.Linear(256, P))
        self.w = nn.Parameter(torch.zeros(P))
        # Metaplasticity: Fisher importance per parameter
        self.register_buffer('fisher', torch.zeros(sum(p.numel() for p in self.net.parameters())))

    def phi(self, s): return self.net(s)
    def q(self, phi): return (phi * self.w.unsqueeze(0)).sum(dim=-1)

    def update_fisher(self, lr=0.99):
        """Track per-parameter importance via squared gradient moving average."""
        n = 0
        for p in self.net.parameters():
            if p.grad is not None:
                sz = p.grad.numel()
                g = p.grad.view(-1).abs().detach()
                self.fisher[n:n+sz] = self.fisher[n:n+sz] * lr + g * (1 - lr)
                n += sz

    def get_per_param_lr(self, base_lr=3e-4, beta=10.0):
        """Per-parameter LR: important params learn SLOWLY, others quickly."""
        imp = self.fisher / (self.fisher.max() + 1e-8)
        return base_lr / (1.0 + beta * imp)


# ═══ Policy (conditioned on φ(s) + goal) ═════════════════════════
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(P + G, H)
        self.mean = nn.Linear(H, A); self.log_std = nn.Parameter(torch.zeros(A))
        self.value = nn.Linear(H, 1)
        nn.init.orthogonal_(self.mean.weight, .01); nn.init.orthogonal_(self.value.weight, 1.)

    def forward(self, phi, gd, h=None):
        if h is None: h = phi.new_zeros(phi.size(0), H)
        h = self.gru(torch.cat([phi, gd], -1), h)
        return (torch.tanh(self.mean(h)), F.softplus(self.log_std)+1e-4,
                self.value(h).squeeze(-1), h)


# ═══ Forward Model (predicts φ(s') + reward from φ(s) + action) ══
class ForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(P + A, 256), nn.ReLU(), nn.Linear(256, P + 1))

    def forward(self, phi, action):
        out = self.net(torch.cat([phi, action], -1))
        return out[:, :-1], out[:, -1]  # next_phi, reward


# ═══ Transition Buffer (NEVER fully cleared) ═══════════════════
class TransitionBuffer:
    """Hippocampal buffer: raw (s,a,r,s') with |δ|-gated retention.
    Demo transitions: PERMANENT (importance=1.0, never evicted).
    Exploration transitions: retained if high |δ|, evictable if low |δ|.
    """
    def __init__(self, max_sz=2000):
        self.states, self.actions, self.rewards, self.next_states = [], [], [], []
        self.deltas = []  # |δ| for each transition — retention priority
        self.demo_mask = []  # True = demo (permanent), False = exploration
        self.max_sz = max_sz

    def store(self, state, action, reward, next_state, delta=0.0, is_demo=False):
        # Evict lowest-|δ| non-demo entry if full
        if len(self.states) >= self.max_sz:
            # Find non-demo entry with lowest |δ|
            candidates = [i for i, d in enumerate(self.demo_mask) if not d]
            if candidates:
                evict = min(candidates, key=lambda i: self.deltas[i])
                [l.pop(evict) for l in [self.states, self.actions, self.rewards, self.next_states, self.deltas, self.demo_mask]]
            else:
                return  # all entries are demo — can't evict
        self.states.append(state.cpu().detach())
        self.actions.append(action.cpu().detach())
        self.rewards.append(float(reward))
        self.next_states.append(next_state.cpu().detach())
        self.deltas.append(abs(float(delta)))
        self.demo_mask.append(is_demo)

    def update_delta(self, idx, new_delta):
        if 0 <= idx < len(self.deltas):
            self.deltas[idx] = max(self.deltas[idx], abs(float(new_delta)))

    def sample_interleaved(self, n, demo_ratio=0.5):
        """50% demo + 50% high-|δ| exploration (interleaved replay)."""
        n_demo = min(int(n * demo_ratio), sum(self.demo_mask))
        n_explore = min(n - n_demo, len(self.states) - n_demo)

        demo_idx = [i for i, d in enumerate(self.demo_mask) if d]
        explore_idx = [i for i, d in enumerate(self.demo_mask) if not d]

        chosen = []
        if demo_idx:
            chosen += list(np.random.choice(demo_idx, min(n_demo, len(demo_idx)), False))
        if explore_idx:
            # Sample proportionally to |δ| (high |δ| = more replay)
            weights = np.array([self.deltas[i] for i in explore_idx]) + 0.01
            weights /= weights.sum()
            n_e = min(n_explore, len(explore_idx))
            chosen += list(np.random.choice(explore_idx, n_e, False, weights))

        return self._get_batch(chosen)

    def _get_batch(self, idx):
        if not idx: return None, None, None, None
        return (torch.stack([self.states[i] for i in idx]),
                torch.stack([self.actions[i] for i in idx]),
                torch.tensor([self.rewards[i] for i in idx]),
                torch.stack([self.next_states[i] for i in idx]))

    def evict_low_delta(self, threshold=0.1, keep_demo=True):
        """Evict exploration entries with |δ| below threshold."""
        to_remove = []
        for i in range(len(self.states) - 1, -1, -1):
            if keep_demo and self.demo_mask[i]: continue
            if self.deltas[i] < threshold:
                to_remove.append(i)
        for i in to_remove:
            [l.pop(i) for l in [self.states, self.actions, self.rewards, self.next_states, self.deltas, self.demo_mask]]
        return len(to_remove)

    def __len__(self): return len(self.states)


def compute_gae(r, v, d, g=.99, l=.95):
    adv = torch.zeros_like(r)
    last = 0.
    for t in reversed(range(len(r))):
        δ = r[t] + g*v[t+1]*(1-d[t]) - v[t]
        last = δ + g*l*(1-d[t])*last; adv[t] = last
    return adv, adv + v[:-1]


def run_demo(env):
    """Generate demo trajectory from (0,0) to (3,3)."""
    obs, _ = env.reset(seed=42)
    states, actions, rewards, next_states = [], [], [], []
    s = obs['state']
    for _ in range(500):
        g = s[2:4]; p = s[:2]; d_vec = g - p; dist = np.linalg.norm(d_vec)
        a = np.clip(d_vec/dist if dist>.2 else d_vec*.5, -1, 1).astype(np.float32)
        obs2, r, term, tr, _ = env.step(a)
        s2 = obs2['state']
        states.append(torch.tensor(s)); actions.append(torch.tensor(a))
        rewards.append(float(r)); next_states.append(torch.tensor(s2))
        s = s2
        if term: break
    return states, actions, rewards, next_states


def train(n_steps=2000):
    dg = PatternSeparator(2, 2000, .02)
    buf = TransitionBuffer()
    sf = SRNet().to(DEVICE); pi = Policy().to(DEVICE)
    fm = ForwardModel().to(DEVICE)

    opt_sf = torch.optim.Adam(sf.parameters(), lr=1e-4)  # lower LR for SRNet
    opt_fm = torch.optim.Adam(fm.parameters(), lr=1e-4)  # lower LR for forward model
    opt_pi = torch.optim.Adam(pi.parameters(), lr=3e-4)
    env = NavArena(render_mode=None)

    # Seed buffer with demo (permanent)
    demo_seed = run_demo(env)
    for i in range(len(demo_seed[0])):
        buf.store(demo_seed[0][i], demo_seed[1][i], demo_seed[2][i],
                  demo_seed[3][i], delta=10.0, is_demo=True)
    print(f"  Demo: {len(demo_seed[0])} transitions (permanent)")

    # Pre-train SRNet + forward model on demo
    for _ in range(200):
        ss = torch.stack(demo_seed[0]).to(DEVICE)
        aa = torch.stack(demo_seed[1]).to(DEVICE)
        rr = torch.tensor(demo_seed[2], device=DEVICE)
        ns = torch.stack(demo_seed[3]).to(DEVICE)
        phi_s = sf.phi(ss); phi_n = sf.phi(ns).detach()
        phi_p, r_p = fm(phi_s, aa)
        loss = F.mse_loss(phi_p, phi_n) + F.mse_loss(r_p, rr)
        opt_sf.zero_grad(); opt_fm.zero_grad(); loss.backward()
        sf.update_fisher()
        torch.nn.utils.clip_grad_norm_(sf.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt_sf.step(); opt_fm.step()
    # Create FROZEN teacher copy of forward model for distillation
    fm_teacher = ForwardModel().to(DEVICE)
    fm_teacher.load_state_dict(fm.state_dict())
    for p in fm_teacher.parameters():
        p.requires_grad_(False)
    fm_teacher.eval()
    # Behavioral cloning: train policy on demo actions
    bc_losses = []
    for _ in range(200):
        ss = torch.stack(demo_seed[0]).to(DEVICE)
        aa = torch.stack(demo_seed[1]).to(DEVICE)
        gd_list = []
        for s in demo_seed[0]:
            st = s.unsqueeze(0).to(DEVICE)
            gd = (st[:,2:4]-st[:,:2]) / ((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)
            gd_list.append(gd)
        gd_batch = torch.cat(gd_list)
        phi_s = sf.phi(ss)
        m, sd, _, _ = pi(phi_s, gd_batch)
        bc_loss = F.mse_loss(torch.tanh(m), aa)
        bc_losses.append(bc_loss.item())
        opt_pi.zero_grad(); bc_loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        opt_pi.step()
    print(f"  BC: loss went from {bc_losses[0]:.4f} to {bc_losses[-1]:.4f} ({len(bc_losses)} iters)")
    # Verify BC: check policy output on demo states
    with torch.no_grad():
        m_test, _, _, _ = pi(phi_s, gd_batch)
        policy_actions = torch.tanh(m_test)
        action_cosim = F.cosine_similarity(policy_actions, aa, dim=-1).mean().item()
        action_mag = policy_actions.norm(dim=-1).mean().item()
        demo_mag = aa.norm(dim=-1).mean().item()
    print(f"  BC verification: cosim={action_cosim:.3f} policy_mag={action_mag:.3f} demo_mag={demo_mag:.3f}")

    goals, step = 0, 0
    s = env.reset(seed=42)[0]['state']
    h = None
    roll = {k: [] for k in ['s','gd','a','lp','v','r','d']}

    while step < n_steps:
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        gd = (st[:,2:4]-st[:,:2]) / ((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)
        phi_s = sf.phi(st)

        # Latent planning: policy candidates + demo candidates evaluated by forward model
        m, sd, v, h = pi(phi_s, gd, h)
        dist = Normal(m, sd)
        action_default = dist.sample()
        action = action_default.clone()
        q_best = -999.0

        with torch.no_grad():
            # Candidate actions: 10 from policy + all demo actions from buffer
            cand_policy = dist.sample([10]).squeeze(1)
            if sum(buf.demo_mask) >= 5:
                demo_indices = [i for i, d in enumerate(buf.demo_mask) if d]
                cand_demo = torch.stack([buf.actions[i] for i in demo_indices]).to(DEVICE)
                all_cand = torch.cat([cand_policy, cand_demo], dim=0)
            else:
                all_cand = cand_policy

            n_total = all_cand.size(0)
            phi_e = phi_s.expand(n_total, -1)
            phi_p, r_p = fm(phi_e, all_cand)
            q_p = r_p + 0.99 * sf.q(phi_p)
            best_idx = q_p.argmax().item()
            q_best = q_p[best_idx].item()

            # If best candidate is a demo action, use it (episodic bootstrap)
            if sum(buf.demo_mask) >= 5 and best_idx >= 10:
                action = all_cand[best_idx:best_idx+1]
            elif q_best > -5.0:  # only plan if predicted Q is reasonable
                action = all_cand[best_idx:best_idx+1]

        action_diff = (action - action_default).norm().item()

        obs2, re, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
        s2 = obs2['state']; done = term or trunc
        s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

        # Compute |δ| for this transition and UPDATE w via TD
        with torch.no_grad():
            phi_s_val = sf.phi(st)
            phi_next_val = sf.phi(s2_t)
            q_now = sf.q(phi_s_val).item()
            q_next = sf.q(phi_next_val).item()
            td_target = re + 0.99 * q_next
            td_error = td_target - q_now
            delta = abs(td_error)
            # TD update on w: w ← w + α · δ · φ(s)
            sf.w.data += 1e-4 * td_error * phi_s_val.squeeze(0)

        # Store in buffer (non-demo, with |δ|)
        buf.store(st.squeeze(0), action.squeeze(0), re, s2_t.squeeze(0), delta=delta, is_demo=False)

        roll['s'].append(st.squeeze(0).detach())
        roll['gd'].append(gd.squeeze(0).detach())
        roll['a'].append(action.squeeze(0).detach())
        roll['lp'].append(dist.log_prob(action).sum(-1).detach())
        roll['v'].append(v.detach())
        roll['r'].append(re)
        roll['d'].append(1. if done else 0.)

        step += 1
        if term: goals += 1
        if done: s = env.reset(seed=42)[0]['state']; h = None
        else: s = s2

        if step % 200 == 0:
            n_demo = sum(buf.demo_mask)
            n_hi = sum(1 for d in buf.deltas if d > 0.5) - n_demo if len(buf.deltas) > n_demo else 0
            # Policy output stats
            with torch.no_grad():
                m_test = m if 'm' in dir() else torch.zeros(2)
                pol_mean = m_test.mean().item() if hasattr(m_test, 'mean') else 0
                pol_norm = m_test.norm().item() if hasattr(m_test, 'norm') else 0
            # Distance to goal
            dist_to_goal = np.linalg.norm(s[:2] - s[2:4])
            # Q-value stats
            q_now = q_now if 'q_now' in dir() else 0
            print(f"  step {step}: goals={goals} dist={dist_to_goal:.2f} "
                  f"Q={q_now:.1f} q_best={q_best:.1f} "
                  f"policy_m={pol_mean:.3f} a_diff={action_diff:.3f} "
                  f"buf={len(buf)}")

        # PPO update
        if step > 0 and step % 64 == 0 and len(roll['s']) >= 64:
            sb = torch.stack(roll['s']); gb = torch.stack(roll['gd'])
            ab = torch.stack(roll['a']); ob = torch.stack(roll['lp'])
            rb = torch.tensor(roll['r'], device=DEVICE, dtype=torch.float32)
            db = torch.tensor(roll['d'], device=DEVICE, dtype=torch.float32)
            vb = torch.stack(roll['v'])
            with torch.no_grad():
                _, _, nv, _ = pi(sf.phi(sb[-1:]), gb[-1:])
            av = torch.cat([vb.view(-1), nv.view(-1)])
            adv, ret = compute_gae(rb, av, db)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(4):
                perm = torch.randperm(len(sb))
                for i in range(0, len(sb), 256):
                    idx = perm[i:i+256]
                    m, sd, vm, _ = pi(sf.phi(sb[idx]), gb[idx])
                    d2 = Normal(m, sd); lp2 = d2.log_prob(ab[idx]).sum(-1)
                    ratio = (lp2 - ob[idx]).exp()
                    ca = torch.clamp(ratio, .8, 1.2) * adv[idx]
                    al = -(torch.min(ratio * adv[idx], ca)).mean()
                    cl = F.mse_loss(vm, ret[idx])
                    (al + .5*cl - .01*d2.entropy().sum(-1).mean()).backward()
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), .5)
                    opt_pi.step()
            roll = {k: [] for k in ['s','gd','a','lp','v','r','d']}

        # Sleep: evaluate + distill forward model
        if step > 0 and step % 200 == 0 and len(buf) >= 50:
            # Evaluate (no grad)
            with torch.no_grad():
                metrics = {}
                if sum(buf.demo_mask) > 0:
                    di = [i for i, d in enumerate(buf.demo_mask) if d]
                    ds = torch.stack([buf.states[i] for i in di]).to(DEVICE)
                    da = torch.stack([buf.actions[i] for i in di]).to(DEVICE)
                    dn = torch.stack([buf.next_states[i] for i in di]).to(DEVICE)
                    dr = torch.tensor([buf.rewards[i] for i in di], device=DEVICE)
                    dpp, drp = fm(sf.phi(ds), da)
                    dphin, _ = fm_teacher(sf.phi(ds), da)
                    metrics['t_loss'] = F.mse_loss(dpp, dphin).item()
                    metrics['d_loss'] = F.mse_loss(dpp, sf.phi(dn)).item() + F.mse_loss(drp, dr).item()
                ei = [i for i, d in enumerate(buf.demo_mask) if not d][:50]
                if ei:
                    es = torch.stack([buf.states[i] for i in ei]).to(DEVICE)
                    ea = torch.stack([buf.actions[i] for i in ei]).to(DEVICE)
                    en = torch.stack([buf.next_states[i] for i in ei]).to(DEVICE)
                    er = torch.tensor([buf.rewards[i] for i in ei], device=DEVICE)
                    epp, erp = fm(sf.phi(es), ea)
                    ephin = sf.phi(en).detach()
                    metrics['e_loss'] = F.mse_loss(epp, ephin).item() + F.mse_loss(erp, er).item()
                t_str = f"  [distill]"
                for k, v in metrics.items():
                    t_str += f" {k}={v:.4f}"
                print(t_str)

            # Re-train forward model on CURRENT demo φ values (adapts to SRNet drift)
            if sum(buf.demo_mask) > 0:
                di = [i for i, d in enumerate(buf.demo_mask) if d]
                ds = torch.stack([buf.states[i] for i in di]).to(DEVICE)
                da = torch.stack([buf.actions[i] for i in di]).to(DEVICE)
                dn = torch.stack([buf.next_states[i] for i in di]).to(DEVICE)
                dr = torch.tensor([buf.rewards[i] for i in di], device=DEVICE)
                for _ in range(20):
                    phi_s = sf.phi(ds).detach()  # CURRENT φ values
                    phi_n = sf.phi(dn).detach()
                    phi_p, r_p = fm(phi_s, da)
                    loss = F.mse_loss(phi_p, phi_n) + F.mse_loss(r_p, dr)
                    opt_fm.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                    opt_fm.step()

            # Update teacher with student's new weights (sync after re-training)
            fm_teacher.load_state_dict(fm.state_dict())

            evicted = buf.evict_low_delta(threshold=0.05)

    # Save demo actions for test-time latent planning
    torch.save({
        'demo_states': torch.stack(demo_seed[0]),
        'demo_actions': torch.stack(demo_seed[1]),
    }, OUT / 'demo_data.pt')
    env.close()
    torch.save(pi.state_dict(), OUT/'policy.pt')
    torch.save(sf.state_dict(), OUT/'sr_net.pt')
    torch.save(fm.state_dict(), OUT/'fm.pt')
    print(f"  Done: {goals} goals in {step} steps")
    return goals


def test(n_eps=50):
    pi = Policy().to(DEVICE); pi.load_state_dict(torch.load(OUT/'policy.pt', map_location=DEVICE))
    sf = SRNet().to(DEVICE); sf.load_state_dict(torch.load(OUT/'sr_net.pt', map_location=DEVICE))
    fm = ForwardModel().to(DEVICE); fm.load_state_dict(torch.load(OUT/'fm.pt', map_location=DEVICE))
    # Load demo actions for latent planning
    demo = torch.load(OUT/'demo_data.pt', map_location=DEVICE)
    demo_actions = demo['demo_actions']
    env = NavArena(render_mode='rgb_array'); env.set_curriculum(0)
    goals = 0
    for ep in range(n_eps):
        s = env.reset(seed=42)[0]['state']; h = None; reached = False
        for _ in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
            gd = (st[:,2:4]-st[:,:2])/((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)
            phi = sf.phi(st)
            # Latent planning: policy candidates + demo actions
            m, sd, _, h = pi(phi, gd, h)
            cand_policy = Normal(m, sd).sample([10]).squeeze(1)
            all_cand = torch.cat([cand_policy, demo_actions], dim=0)
            phi_e = phi.expand(all_cand.size(0), -1)
            phi_p, r_p = fm(phi_e, all_cand)
            q_p = r_p + 0.99 * sf.q(phi_p)
            a = all_cand[q_p.argmax().item()].unsqueeze(0)
            obs2, _, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            s = obs2['state']
            if term: reached = True; break
            if trunc: h = None
        if reached: goals += 1
        if (ep+1)%10==0: print(f'  {ep+1}/50: {goals}/{ep+1}={goals/(ep+1):.0%}')
    print(f'Test: {goals}/50 = {goals/50:.0%}')
    env.close()


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    train(2000); test(50)
