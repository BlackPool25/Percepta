"""Brain-like architecture: Hippocampus + Neocortex + BG gate + PFC working memory.

Systems (no conflicts):
  Hippocampus:  QMemory — one-shot Q-value storage, content-addressable retrieval
  Neocortex:    SRNet φ^T·w — slow generalization, predictive coding
  BG gate:      Binary Go/NoGo — decides WHEN to update PFC working memory
  PFC:          GRU hidden state — maintains goal representation

Fix 1: Binary BG gate (Go/NoGo, not continuous blend)
Fix 2: RPE trains the BG gate (dopamine → gate, not w)
Fix 3: Sleep consolidation (hippocampus → neocortex replay)
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from hopfield_memory import PatternSeparator
from env_nav import NavArena

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('results/sr'); OUT.mkdir(parents=True, exist_ok=True)
S, P, H, G, A = 12, 64, 64, 2, 2

# ═══ Neocortex: SR Network ═══════════════════════════════════════
class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(S, 128), nn.ReLU(), nn.Linear(128, P))
        self.w = nn.Parameter(torch.zeros(P))
    def phi(self, s): return self.net(s)
    def q(self, phi): return (phi * self.w.unsqueeze(0)).sum(dim=-1)

# ═══ PFC + BG: Working Memory + Binary Gate ═════════════════════
class Policy(nn.Module):
    """PFC (GRU) maintains working memory. BG (gate) decides WHEN to update.
    Gate is BINARY: Go (update) or NoGo (protect). Trained by dopamine RPE."""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(S + G, H)
        self.gate = nn.Sequential(nn.Linear(H + 1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.mean = nn.Linear(H, A)
        self.log_std = nn.Parameter(torch.zeros(A))
        self.value = nn.Linear(H, 1)
        nn.init.orthogonal_(self.mean.weight, 0.01)
        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, x, h=None, rpe=None):
        if h is None: h = x.new_zeros(x.size(0), H)
        hn = self.gru(x, h)                          # PFC proposes new state
        r = rpe if rpe is not None else x.new_zeros(x.size(0), 1)
        g = torch.sigmoid(self.gate(torch.cat([h.detach(), r], -1)))
        # FIX 1: Binary gate (Go/NoGo), not continuous blend
        ho = torch.where(g > 0.5, hn, h)             # BG decides: update or protect
        return (torch.tanh(self.mean(ho)),
                F.softplus(self.log_std) + 1e-4,
                self.value(ho).squeeze(-1), ho, g.squeeze(-1))

# ═══ Hippocampus: QMemory ════════════════════════════════════════
class QMemory:
    """Hippocampus: one-shot (state, action, Q) storage. Direct action retrieval."""
    def __init__(self, max_sz=2000, beta=2.0):
        self.keys, self.states, self.actions, self.qs, self.imp = [], [], [], [], []
        self.max_sz, self.beta = max_sz, beta
        self.last_idx = None

    def store(self, key, state, action, q_val):
        if len(self.keys) >= self.max_sz:
            idx = int(np.argmin(self.imp))
            self.keys.pop(idx); self.states.pop(idx)
            self.actions.pop(idx); self.qs.pop(idx); self.imp.pop(idx)
        self.keys.append(key.cpu().detach())
        self.states.append(state.cpu().detach())
        self.actions.append(action.cpu().detach())
        self.qs.append(float(q_val))
        self.imp.append(0.5)

    def retrieve(self, q):
        """Returns (state_star, action_star, q_star, rbf_sim, idx)."""
        if not self.keys:
            self.last_idx = None
            return (q.new_zeros(1, S), q.new_zeros(1, A),
                    q.new_zeros(1), 0.0, None)
        Z = torch.stack(self.keys).to(q.device).to(q.dtype)
        logits = self.beta * (q @ Z.T)
        attn = F.softmax(logits, -1)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        Zn = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)
        rbf = torch.exp(-2.0 * (1.0 - (qn @ Zn.T).squeeze(0))).max().item()
        self.last_idx = attn.argmax().item() if attn.max().item() > 0.1 else None
        self._upd_imp(attn.squeeze(0))
        Ss = torch.stack(self.states).to(q.device).to(q.dtype)
        As = torch.stack(self.actions).to(q.device).to(q.dtype)
        Qs = torch.tensor(self.qs, device=q.device, dtype=q.dtype).unsqueeze(-1)
        best_idx = attn.argmax(dim=-1).item()
        return (attn @ Ss, As[best_idx].unsqueeze(0),  # use BEST action, not blend
                (attn @ Qs).squeeze(-1), rbf, self.last_idx)

    def td_update(self, idx, td_error, lr=0.3):
        if idx is not None and idx < len(self.qs):
            self.qs[idx] += lr * td_error

    def can_destabilize(self, idx, abs_rpe, thresh=0.3):
        return idx is not None and abs_rpe > thresh

    def sample(self, n, t=0.5):
        if not self.states: return None, None, None, None
        imp = np.clip(np.array(self.imp, np.float64), 0.01, 1.0)
        p = imp ** (1/t); p /= p.sum() + 1e-10
        idx = np.random.choice(len(self.states), min(n, len(self.states)), False, p)
        return (torch.stack([self.keys[i] for i in idx]),
                torch.stack([self.states[i] for i in idx]),
                torch.stack([self.actions[i] for i in idx]),
                torch.tensor([self.qs[i] for i in idx]))

    def sample_trajectory(self, traj_len=10):
        """Sample a CONTIGUOUS trajectory from memory for structured replay."""
        if len(self.states) < traj_len + 1:
            return None, None, None, None
        start = np.random.randint(0, len(self.states) - traj_len)
        sl = slice(start, start + traj_len)
        return (torch.stack([self.keys[i] for i in range(start, start+traj_len)]),
                torch.stack([self.states[i] for i in range(start, start+traj_len)]),
                torch.stack([self.actions[i] for i in range(start, start+traj_len)]),
                torch.tensor([self.qs[i] for i in range(start, start+traj_len)]))

    def retrieve_with_pfc(self, pos_key, pfc_h, goal_dir, goal_weight=0.3):
        """PFC-modulated retrieval: position + goal context."""
        if not self.keys: return self.retrieve(pos_key)
        Z = torch.stack(self.keys).to(pos_key.device).to(pos_key.dtype)
        Ss = torch.stack(self.states).to(pos_key.device).to(pos_key.dtype)
        As = torch.stack(self.actions).to(pos_key.device).to(pos_key.dtype)
        Qs = torch.tensor(self.qs, device=pos_key.device, dtype=pos_key.dtype).unsqueeze(-1)

        # Position-based similarity (base retrieval)
        logits_pos = self.beta * (pos_key @ Z.T)

        # PFC goal modulation: prefer patterns with similar goal direction
        stored_goals = Ss[:, 2:4] - Ss[:, :2]  # goal directions from stored states
        stored_goals = stored_goals / (stored_goals.norm(dim=-1, keepdim=True) + 1e-8)
        goal_sim = (goal_dir @ stored_goals.T).squeeze(0)
        logits = logits_pos + goal_weight * goal_sim

        attn = F.softmax(logits, -1)
        qn = pos_key / (pos_key.norm(dim=-1, keepdim=True) + 1e-8)
        Zn = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)
        rbf = torch.exp(-2.0 * (1.0 - (qn @ Zn.T).squeeze(0))).max().item()

        self.last_idx = attn.argmax().item() if attn.max().item() > 0.1 else None
        self._upd_imp(attn.squeeze(0))
        best_idx = attn.argmax(dim=-1).item()
        return (attn @ Ss, As[best_idx].unsqueeze(0),
                (attn @ Qs).squeeze(-1), rbf, self.last_idx)

    def _upd_imp(self, w):
        ws = w.detach().cpu().numpy()
        for i, v in enumerate(ws):
            if v > 0.05: self.imp[i] = min(self.imp[i] + 0.05, 1.0)
    def decay_imp(self, f=0.99):
        for i in range(len(self.imp)): self.imp[i] = max(self.imp[i]*f, 0.01)
    def __len__(self): return len(self.keys)


def run_demo(mem, dg):
    """Seed hippocampus with one successful trajectory."""
    env = NavArena(render_mode='rgb_array')
    obs, _ = env.reset(seed=42); data = []
    for _ in range(500):
        g = obs['state'][2:4]; p = obs['state'][:2]; d = g - p; dist = np.linalg.norm(d)
        a = np.clip(d/dist if dist>0.2 else d*0.5, -1, 1).astype(np.float32)
        obs2, r, term, tr, _ = env.step(a)
        data.append({'s': obs['state'].copy(), 'r': r}); obs = obs2
        if term: break
    env.close()
    qs = [0.0]*len(data); c = 0.0
    for i in range(len(data)-1, -1, -1): c = data[i]['r'] + 0.99*c; qs[i] = c
    # Re-run demo to capture actions
    env2 = NavArena(render_mode='rgb_array')
    obs2, _ = env2.reset(seed=42)
    for i in range(len(data)):
        g = obs2['state'][2:4]; p = obs2['state'][:2]; d = g - p; dist = np.linalg.norm(d)
        act = np.clip(d/dist if dist>0.2 else d*0.5, -1, 1).astype(np.float32)
        z = dg(torch.tensor(p, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        mem.keys.append(z.cpu()); mem.states.append(torch.tensor(obs2['state']))
        mem.actions.append(torch.tensor(act))
        mem.qs.append(qs[i]); mem.imp.append(1.0)
        obs2, _, term, tr, _ = env2.step(act)
        if term: break
    env2.close()
    print(f"  Demo: {len(data)} steps, Q=[{qs[-1]:.1f}, {qs[0]:.1f}]")


def train(n_steps=500):
    dg = PatternSeparator(2, 2000, 0.02)
    mem = QMemory(); sf = SRNet().to(DEVICE)
    pi = Policy().to(DEVICE)
    fm = nn.Sequential(nn.Linear(P + A + G, 128), nn.ReLU(), nn.Linear(128, P)).to(DEVICE)
    opt = torch.optim.Adam(list(pi.parameters()) + list(sf.parameters()) + list(fm.parameters()), lr=3e-4)
    env = NavArena(render_mode='rgb_array'); env.set_curriculum(0)

    run_demo(mem, dg)
    print(f"\nTraining {n_steps} steps...")
    obs, _ = env.reset(seed=42); s = obs['state']
    h = None; goals = 0
    roll = {k: [] for k in ['s','gd','a','lp','v','r','d','zk','gt']}

    for step in range(n_steps):
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        gd = (st[:,2:4]-st[:,:2]) / ((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)
        z = dg(st[:,:2])

        # Memory retrieval with PFC gating (goal context from working memory)
        with torch.no_grad():
            if h is not None:
                ss, a_star, q_mem, rbf, ridx = mem.retrieve_with_pfc(z, h, gd)
            else:
                ss, a_star, q_mem, rbf, ridx = mem.retrieve(z)
            q_mem_v = q_mem.item()
            q_par = sf.q(sf.phi(st)).item()

        # Episodic control: stored action when confident + high Q
        blend = 0.8 if rbf > 0.3 else (0.3 if rbf > 0.1 else 0.0)
        q_used = blend * q_mem_v + (1-blend) * q_par
        pi_in = torch.cat([st.squeeze(0), gd.squeeze(0)], dim=-1).unsqueeze(0)
        m, sd, v, h, gate = pi(pi_in, h)
        use_episodic = rbf > 0.3 and q_mem_v > 10.0
        a = a_star if use_episodic else Normal(m, sd).sample()
        a = a.detach()

        obs2, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
        s2 = obs2['state']
        nxt = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

        # Dopamine RPE: trains the BG gate (when to update PFC working memory)
        _, _, q_next_mem, _, _ = mem.retrieve(dg(nxt[:,:2]))
        td_target = re + 0.99 * q_next_mem.item()
        td_error = td_target - q_used
        gate_target = 1.0 if td_error > 0 else 0.0
        gate_loss = (gate - gate_target) ** 2

        # TD on w: separate from gate — standard value learning
        q_next_par = sf.q(sf.phi(nxt)).item()
        td_par = re + 0.99 * q_next_par - q_par
        sf.w.data += 1e-4 * td_par * sf.phi(st).squeeze(0).detach()

        # Selective destabilization on high |δ|
        if mem.can_destabilize(ridx, abs(td_error), 0.5):
            mem.td_update(ridx, td_error * 0.3)

        # Store novel experiences (with action for direct retrieval)
        if rbf < 0.1 or abs(td_error) > 0.5:
            mem.store(z.squeeze(0), st.squeeze(0), a.squeeze(0), td_target)

        mem.decay_imp()
        if term: goals += 1
        if term or trunc:
            obs, _ = env.reset(seed=42); s = obs['state']; h = None
        else:
            s = s2

        # Gate loss trains the BG gate
        gate_opt = torch.optim.SGD(pi.gate.parameters(), lr=1e-3)
        gate_opt.zero_grad(); gate_loss.backward(); gate_opt.step()

        if step % 50 == 0:
            print(f"  step {step:3d}: ext={re:.3f} Qm={q_mem_v:.1f} Qp={q_par:.1f} "
                  f"RPE={td_error:.2f} gate={gate.item():.2f} goals={goals} mem={len(mem)}")

        # Sleep: structured trajectory replay (forward + backward)
        if step > 0 and step % 100 == 0:
            # Forward replay: predict next φ from current φ + action
            for _ in range(5):
                traj = mem.sample_trajectory(10)
                if traj and traj[0] is not None:
                    _, ss, aa, qq = traj
                    ss, aa, qq = ss.to(DEVICE), aa.to(DEVICE), qq.to(DEVICE)
                    phi = sf.phi(ss)
                    # Backward: Q prediction (standard)
                    q_pred = sf.q(phi)
                    q_loss = F.mse_loss(q_pred, qq)
                    # Forward: predict φ_{t+1} from φ_t + a_t
                    phi_now, phi_next = phi[:-1], phi[1:].detach()
                    a_now = aa[:-1]
                    gd_now = (ss[:-1, 2:4] - ss[:-1, :2])
                    gd_now = gd_now / (gd_now.norm(dim=-1, keepdim=True) + 1e-8)
                    # Forward model: predict φ(s') from [φ(s), a, gd]
                    fm_in = torch.cat([phi_now, a_now, gd_now], dim=-1)
                    fm_pred = fm(fm_in)
                    fm_loss = F.mse_loss(fm_pred, phi_next)
                    loss = q_loss + 0.1 * fm_loss
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(sf.parameters(), 1.0)
                    opt.step()

            # Random high-Q consolidation (selective: only high-Q patterns)
            ks, ss, aa, qq = mem.sample(64)
            if ks is not None:
                ss, qq = ss.to(DEVICE), qq.to(DEVICE)
                # Only train on high-Q patterns (selective consolidation)
                high_q_mask = qq > qq.median()
                if high_q_mask.any():
                    q_pred = sf.q(sf.phi(ss[high_q_mask]))
                    loss = F.mse_loss(q_pred, qq[high_q_mask])
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(sf.parameters(), 1.0)
                    opt.step()

    print(f"\n  Done: {goals} goals in {n_steps} steps")
    env.close()
    torch.save(pi.state_dict(), OUT / 'policy.pt')
    return goals


def test(n_eps=20, dg=None, mem=None):
    pi = Policy().to(DEVICE)
    pi.load_state_dict(torch.load(OUT / 'policy.pt', map_location=DEVICE))
    # Load or create memory + DG for full system test
    if dg is None: dg = PatternSeparator(2, 2000, 0.02)
    if mem is None: mem = QMemory(); run_demo(mem, dg)  # seed fresh memory
    env = NavArena(render_mode='rgb_array'); env.set_curriculum(0)
    goals = 0
    for ep in range(n_eps):
        obs, _ = env.reset(seed=42); s = obs['state']; h = None; reached = False
        for _ in range(500):
            with torch.no_grad():
                st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
                gd = (st[:,2:4]-st[:,:2])/((st[:,2:4]-st[:,:2]).norm(dim=-1,keepdim=True)+1e-8)
                z = dg(st[:,:2])
                ss, a_star, q_mem, rbf, ridx = mem.retrieve(z)
                # Use episodic action when confident + high Q
                use_epi = rbf > 0.3 and q_mem.item() > 10.0
                if use_epi:
                    a = a_star
                else:
                    pi_in = torch.cat([st.squeeze(0), gd.squeeze(0)], dim=-1).unsqueeze(0)
                    m, sd, _, h, _ = pi(pi_in, h)
                    a = Normal(m, sd).sample()
                obs2, _, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
                s = obs2['state']
                if term: reached = True; break
                if trunc: break
        if reached: goals += 1
        print(f"  Ep {ep+1}: {'GOAL' if reached else 'fail'}")
    print(f"  Test: {goals}/{n_eps} = {goals/n_eps:.0%}")
    env.close()


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    train(500); test(20)
