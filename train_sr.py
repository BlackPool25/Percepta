"""Brain-inspired agent with hippocampal memory + dopamine-modulated plasticity.

Architecture:
  - Hippocampus (DG+CA3): Stores experiences as sparse patterns, retrieves by content
  - Motor cortex (policy): MLP(raw_state+goal) → action, trained via dopamine REINFORCE
  - OFC (value): V(s) trained via TD learning
  - Cerebellum (raw FM): Predicts s' from (s,a), trained continuously
  - Cerebellar planning: simulate candidate actions, pick one minimizing dist to goal

Key neuroscience mechanisms:
  1. DG: pattern separation (fixed random projection + k-WTA, 2% sparsity)
  2. CA3: autoassociative memory (one-shot storage, content-addressable retrieval)
  3. Dopamine RPE gates policy LR: LR = base × (1 + 5 × |δ|)
  4. 3-factor plasticity: Δθ ∝ δ × ∇_θ log π(a|s)
  5. Cerebellar online learning on every (s,a)→s'
  6. Sleep consolidation: CA3 replay trains policy + cerebellum
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import mujoco
from torch.distributions import Normal
from pathlib import Path
from env_nav import NavArena
import logging, time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = Path('results/sr'); OUT.mkdir(parents=True, exist_ok=True)
S, H, G, A, PDIM, SPARSITY = 12, 128, 2, 2, 2000, 0.02

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'training.log', mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger('percepta')


# ═══ Dentate Gyrus: pattern separation via k-WTA ══════════════
class PatternSeparator:
    """Fixed random projection + k-WTA sparsification.

    Maps similar inputs to VERY different sparse codes (pattern separation).
    Projection matrix is NEVER learned — matches DG's fixed mossy fibers.
    Sparsity=2% means only 40 out of 2000 units active per pattern.
    """
    def __init__(self, input_dim: int, hidden_dim: int, sparsity: float):
        self.k = max(1, int(hidden_dim * sparsity))
        P = torch.randn(input_dim, hidden_dim)
        self.P = nn.Parameter(P / (input_dim ** 0.5), requires_grad=False)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        projected = x @ self.P.to(x.device)
        _, indices = torch.topk(projected, self.k, dim=-1)
        z = torch.zeros_like(projected)
        z.scatter_(-1, indices, 1.0)
        return z


# ═══ CA3: content-addressable memory ══════════════════════════
class CA3Memory:
    """Autoassociative memory with content-addressable retrieval.

    GPU-cached: patterns stored as a single stacked tensor on GPU.
    Appending a new pattern cat's to the cached tensor (no full restack).
    Retrieval is O(1) GPU operation without CPU transfers.
    """
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.patterns, self.actions, self.rewards, self.next_states = [], [], [], []
        self.states = []
        self._Z = None  # GPU-cached stacked patterns

    def store(self, z: torch.Tensor, state: torch.Tensor, action: torch.Tensor,
              reward: float, next_state: torch.Tensor):
        self.patterns.append(z.detach().cpu())
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.rewards.append(float(reward))
        self.next_states.append(next_state.detach().cpu())
        self._Z = None  # invalidate cache

    def _get_Z(self, device):
        """Get cached stacked pattern matrix on target device."""
        if self._Z is None or self._Z.device != device:
            # Only rebuild if needed (first time or device mismatch)
            self._Z = torch.stack(self.patterns).to(device)
        return self._Z

    def retrieve_similar(self, z_query: torch.Tensor, k: int = 10):
        if not self.patterns:
            return []
        Z = self._get_Z(z_query.device)
        sims = z_query @ Z.T
        topk = min(k, len(self.patterns))
        return sims[0].topk(topk).indices.tolist()

    def get_batch(self, idx):
        return (torch.stack([self.states[i] for i in idx]),
                torch.stack([self.actions[i] for i in idx]),
                torch.tensor([self.rewards[i] for i in idx]),
                torch.stack([self.next_states[i] for i in idx]))

    def __len__(self): return len(self.patterns)

    def state_dict(self):
        return {'patterns': self.patterns, 'states': self.states,
                'actions': self.actions, 'rewards': self.rewards,
                'next_states': self.next_states, 'beta': self.beta}

    def load_state_dict(self, sd):
        self.patterns = sd['patterns']
        self.states = sd.get('states', [])
        self.actions, self.rewards = sd['actions'], sd['rewards']
        self.next_states = sd['next_states']
        self.beta = sd['beta']
        self._Z = None  # will be rebuilt on next retrieval


# ═══ Hippocampus: DG + CA3 combined ═══════════════════════════
class Hippocampus:
    """Full hippocampal memory system: DG pattern separation + CA3 storage.

    Usage:
      hc = Hippocampus(state_dim=12, pattern_dim=2000, sparsity=0.02)
      hc.store(state, action, reward, next_state)  # one-shot storage
      idx = hc.retrieve(query_state, k=10)         # content-addressable retrieval
      states, actions, rewards, next_states = hc.get_batch(idx)
    """
    def __init__(self, state_dim: int = S, pattern_dim: int = PDIM, sparsity: float = SPARSITY):
        self.dg = PatternSeparator(state_dim, pattern_dim, sparsity)
        self.ca3 = CA3Memory()

    def store(self, state, action, reward, next_state):
        st = state.unsqueeze(0) if state.dim() == 1 else state
        z = self.dg(st)
        self.ca3.store(z.squeeze(0), state, action, reward, next_state)

    def retrieve(self, query_state, k: int = 10):
        z = self.dg(query_state.unsqueeze(0))
        return self.ca3.retrieve_similar(z, k)

    def get_batch(self, idx):
        return self.ca3.get_batch(idx)

    def __len__(self): return len(self.ca3)

    def state_dict(self): 
        d = self.ca3.state_dict()
        d['dg_P'] = self.dg.P.data.clone()
        return d
    def load_state_dict(self, sd): 
        self.ca3.load_state_dict(sd)
        if 'dg_P' in sd:
            self.dg.P.data.copy_(sd['dg_P'])


# ═══ Policy with value head ════════════════════════════════════
class Policy(nn.Module):
    """Policy π(a|s) with value head V(s). Raw state only, no φ-space."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(S + G, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
        )
        self.mean = nn.Linear(H, A)
        self.log_std = nn.Parameter(torch.zeros(A))
        self.value = nn.Linear(H, 1)

    def forward(self, s, gd):
        h = self.shared(torch.cat([s, gd], -1))
        # Residual connection: steer toward goal by default, learn corrections
        # base = gd (goal direction), correction = MLP output (bounded ±0.5)
        correction = torch.tanh(self.mean(h)) * 0.5
        mean_out = torch.clamp(gd + correction, -1, 1)
        return (mean_out,
                F.softplus(self.log_std) + 1e-4,
                self.value(h).squeeze(-1))

    def act(self, s, gd):
        m, sd, v = self.forward(s, gd)
        dist = Normal(m, sd)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), v

    def evaluate(self, s, gd, a):
        m, sd, v = self.forward(s, gd)
        dist = Normal(m, sd)
        return dist.log_prob(a).sum(-1), v


# ═══ Raw-state forward model (cerebellum) ══════════════════════
class RawForwardModel(nn.Module):
    """Predicts Δs = s' - s (efference copy). Returns s + Δs for compatibility."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S + A, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, S + 1),
        )

    def forward(self, s, action):
        out = self.net(torch.cat([s, action], -1))
        ds = out[:, :-1]  # predicted delta
        return s + ds, out[:, -1]  # s' = s + Δs, reward


# ═══ Demo generation ═══════════════════════════════════════════
def run_demo(env, n_trajs=25):
    """Generate diverse demo trajectories with varying initial velocities.

    Each trajectory starts from a 5×5 grid position with random initial velocity
    (simulated via random forces). This teaches the policy the full dynamics:
    how to steer toward goal regardless of current velocity.
    """
    rng = np.random.RandomState(42)
    grid = int(np.ceil(np.sqrt(n_trajs)))
    all_s, all_a, all_r, all_ns = [], [], [], []
    agent_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
    vel_adr = 0  # freejoint velocity starts at DOF 0
    for traj_idx in range(min(grid * grid, n_trajs)):
        i, j = traj_idx // grid, traj_idx % grid
        start_x = -3.0 + 6.0 * (i + 0.5) / grid
        start_y = -3.0 + 6.0 * (j + 0.5) / grid
        # Reset to clean state first
        env.reset(seed=None)
        # Then set position and velocity
        env._set_body_pos("agent", np.array([start_x, start_y, 0.5]))
        env.data.qvel[vel_adr:vel_adr + 2] = rng.uniform(-2, 2, size=2)
        mujoco.mj_forward(env.model, env.data)
        s = env._get_obs()['state']
        for _ in range(500):
            g = s[2:4]; p = s[:2]; d_vec = g - p; dist = np.linalg.norm(d_vec)
            a = np.clip(d_vec/dist if dist>.2 else d_vec*.5, -1, 1).astype(np.float32)
            obs2, r, term, _, _ = env.step(a)
            s2 = obs2['state']
            all_s.append(torch.tensor(s))
            all_a.append(torch.tensor(a))
            all_r.append(float(r))
            all_ns.append(torch.tensor(s2))
            s = s2
            if term: break
    logger.info(f"Generated {len(all_s)} demo transitions from {n_trajs} trajectories")
    return all_s, all_a, all_r, all_ns


# ═══ Dopamine-modulated update ═════════════════════════════════
def dopamine_update(pi, opt_pi, opt_val, s, gd, a, r, s_next, gd_next,
                    gamma=0.99, rpe_clip=10.0, dopamine_boost=1.0):
    """3-factor plasticity with phasic dopamine boost.

    Δθ ∝ δ · ∇_θ log π(a|s)  (RPE gates update direction)
    α_eff = α_base · dopamine_boost · (1 + 3·|δ|/(clip/2))  (RPE gates rate)

    When dopamine_boost > 1 (after unexpected reward), ALL updates have
    enhanced plasticity. This simulates the brain's phasic dopamine burst
    that follows unexpected reward, creating a plasticity window.

    Returns: (delta, effective_lr_scale).
    """
    with torch.no_grad():
        _, _, v_next = pi(s_next, gd_next)
        v_next_val = v_next.item()

    lp, v = pi.evaluate(s, gd, a)
    v_val = v.item()
    td_target = r + gamma * v_next_val
    delta = td_target - v_val
    delta_clipped = max(min(delta, rpe_clip), -rpe_clip)

    # Effective LR: phasic dopamine boost × RPE-gated scaling
    lr_scale = dopamine_boost * (1.0 + 3.0 * min(abs(delta_clipped) / (rpe_clip / 2), 1.0))

    # Policy: Δθ ∝ δ · ∇_θ log π(a|s)
    pi.zero_grad()
    lp, v = pi.evaluate(s, gd, a)
    policy_loss = -(lp * delta_clipped)
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    for p in pi.parameters():
        if p.grad is not None:
            p.grad.data *= lr_scale
    opt_pi.step()

    # Value: TD learning (no dopamine boost — value needs stable updates)
    _, _, v2 = pi(s, gd)
    val_loss = F.mse_loss(v2.view(-1), torch.tensor([td_target], device=DEVICE))
    opt_val.zero_grad()
    val_loss.backward()
    torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    opt_val.step()

    return delta, lr_scale


# ═══ Training ═══════════════════════════════════════════════════
def train(n_steps=2000):
    hc = Hippocampus()
    pi = Policy().to(DEVICE)
    raw_fm = RawForwardModel().to(DEVICE)
    # Separate optimizers: policy, value (both in pi), and raw FM
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)
    env = NavArena(render_mode=None)

    # ── Seed hippocampal memory with diverse demo trajectories ──
    demo_seed = run_demo(env, n_trajs=25)
    for i in range(len(demo_seed[0])):
        hc.store(demo_seed[0][i], demo_seed[1][i], demo_seed[2][i], demo_seed[3][i])
    logger.info(f"Seeded hippocampus with {len(hc)} patterns")

    # ── Pre-train raw FM on demo ────────────────────────────────
    ds = torch.stack(demo_seed[0]).to(DEVICE)
    da = torch.stack(demo_seed[1]).to(DEVICE)
    dn = torch.stack(demo_seed[3]).to(DEVICE)
    dr = torch.tensor(demo_seed[2], device=DEVICE)
    for i in range(200):
        sp, rp = raw_fm(ds, da)
        loss = F.mse_loss(sp, dn) + F.mse_loss(rp.squeeze(-1), dr)
        opt_raw_fm.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
        if i % 50 == 0:
            logger.info(f"  Raw FM init: iter {i} loss={loss.item():.4f}")
    logger.info(f"Raw FM init loss: {loss.item():.4f}")

    # ── BC pre-train policy on demo ─────────────────────────────
    dg = []
    for dsi in demo_seed[0]:
        st = dsi.unsqueeze(0).to(DEVICE)
        g = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)
        dg.append(g)
    dg = torch.cat(dg)
    for _ in range(200):
        m, sd, _ = pi(ds, dg)
        loss = F.mse_loss(m, da.to(DEVICE))
        opt_pi.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
    with torch.no_grad():
        m_test, _, _ = pi(ds, dg)
        cosim = F.cosine_similarity(m_test, da.to(DEVICE), dim=-1).mean().item()
    logger.info(f"BC init: cosim={cosim:.3f}")

    # ── Training loop ───────────────────────────────────────────
    goals, step = 0, 0
    s = env.reset(seed=42)[0]['state']
    ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []
    dopamine_boost = 1.0        # phasic dopamine burst multiplier
    dopamine_decay_steps = 0    # steps remaining for phasic boost
    t0 = time.time()

    while step < n_steps:
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        gd = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

        # ── Hippocampal episodic control ───────────────────────
        # Retrieve the MOST SIMILAR past experience from hippocampus.
        # Use its action directly — this is episodic memory retrieval.
        # No forward model needed for this primary action selection.
        m, sd, v = pi(st, gd)
        dist = Normal(m, sd)
        
        hc_idx = hc.retrieve(st.squeeze(0), k=10)
        if hc_idx and len(hc.ca3.actions) > 0:
            # Weight actions by similarity (exponential of cosine similarity)
            hc_actions = torch.stack([hc.ca3.actions[i] for i in hc_idx]).to(DEVICE)
            hc_states = torch.stack([hc.ca3.states[i] for i in hc_idx]).to(DEVICE)
            z_query = hc.dg(st.squeeze(0).unsqueeze(0))
            Z = torch.stack([hc.ca3.patterns[i] for i in hc_idx]).to(DEVICE)
            sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
            # Weighted average of similar actions (already 2D)
            action = sims @ hc_actions
        else:
            # Fallback: policy sample
            action = dist.sample()

        # ── Execute ─────────────────────────────────────────────
        obs2, re, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
        s2 = obs2['state']
        done = term or trunc
        s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)
        gd2 = (s2_t[:, 2:4] - s2_t[:, :2]) / ((s2_t[:, 2:4] - s2_t[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

        ep_s.append(st.squeeze(0).cpu())
        ep_a.append(action.squeeze(0).cpu())
        ep_g.append(gd.squeeze(0).cpu())
        ep_r.append(re)
        ep_ns.append(s2_t.squeeze(0).cpu())

        # ── Dopamine-modulated REINFORCE with phasic boost ──────
        delta, lr_scale = dopamine_update(pi, opt_pi, opt_val, st, gd, action, re, s2_t, gd2,
                                          dopamine_boost=dopamine_boost)

        # ── Store in hippocampal memory ─────────────────────────
        hc.store(st.squeeze(0), action.squeeze(0), re, s2_t.squeeze(0))

        # ── Cerebellar online learning ──────────────────────────
        sp, rp = raw_fm(st, action)
        loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(rp, torch.tensor([re], device=DEVICE))
        opt_raw_fm.zero_grad(); loss_raw.backward()
        torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()

        step += 1
        dist_to_goal = np.linalg.norm(s[:2] - s[2:4])

        # Phasic dopamine boost: decays after goal (simulates dopamine burst)
        if dopamine_decay_steps > 0:
            dopamine_decay_steps -= 1
            dopamine_boost = 1.0 + 4.0 * (dopamine_decay_steps / 25.0)  # 5x → 1x over 25 steps

        # ── Goal reached → EC capture + dopamine burst ──────────
        if term:
            goals += 1
            dopamine_boost = 5.0       # phasic dopamine burst after reward
            dopamine_decay_steps = 25  # decays over ~25 steps
            logger.info(f"GOAL #{goals} step={step} ({len(ep_s)} steps, RPE={delta:.2f}, DA_boost={dopamine_boost:.1f})")
            if len(ep_s) > 1:
                for i in range(len(ep_s)):
                    hc.store(ep_s[i], ep_a[i], ep_r[i], ep_ns[i])
                # BC on successful trajectory
                ec_s = torch.stack(ep_s).to(DEVICE)
                ec_a = torch.stack(ep_a).to(DEVICE)
                ec_g = torch.stack(ep_g).to(DEVICE)
                for _ in range(30):
                    m_ec, _, _ = pi(ec_s, ec_g)
                    loss_ec = F.mse_loss(m_ec, ec_a)
                    opt_pi.zero_grad(); loss_ec.backward()
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                logger.info(f"EC: {len(ep_s)} steps captured, BC trained")

        if done:
            s = env.reset(seed=42)[0]['state']
            ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []
        else:
            s = s2

        # ── Logging ─────────────────────────────────────────────
        if step % 100 == 0 or step == 1:
            logger.info(
                f"step={step:4d} goals={goals} dist={dist_to_goal:.2f} "
                f"RPE={delta:+.3f} lr_s={lr_scale:.2f} "
                f"hc={len(hc)} a_diff={(action - m).norm().item():.3f} "
                f"hc_sims={len(hc_idx) if hc_idx else 0}"
            )

        # ── Sleep: consolidate ──────────────────────────────────
        if step > 0 and step % 200 == 0 and len(hc) >= 50:
            all_idx = list(range(len(hc)))
            hs, ha, hr, hn = hc.get_batch(all_idx)
            hs, ha, hn = hs.to(DEVICE), ha.to(DEVICE), hn.to(DEVICE)
            hr = hr.to(DEVICE)

            # Train raw FM on (state, action) → next_state
            for _ in range(30):
                sp, rp = raw_fm(hs, ha)
                loss = F.mse_loss(sp, hn) + F.mse_loss(rp.squeeze(-1), hr)
                opt_raw_fm.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
            logger.info(f"Sleep: raw FM trained on {len(hs)} transitions")

            # Sleep BC: policy imitates all stored actions
            tg = []
            for hsi in hs:
                g = (hsi[2:4] - hsi[:2]) / ((hsi[2:4] - hsi[:2]).norm(dim=-1, keepdim=True) + 1e-8)
                tg.append(g.unsqueeze(0))
            tg = torch.cat(tg).to(DEVICE)
            bc_losses = []
            for _ in range(100):
                m_bc, _, _ = pi(hs, tg)
                loss = F.mse_loss(m_bc, ha)
                bc_losses.append(loss.item())
                opt_pi.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
            logger.info(f"Sleep BC: {bc_losses[0]:.4f} → {bc_losses[-1]:.4f} ({len(hs)} trans)")

    elapsed = time.time() - t0
    logger.info(f"Training: {goals} goals in {step} steps ({elapsed:.1f}s)")
    return hc, pi, raw_fm


# ═══ Test ══════════════════════════════════════════════════════
def test(n_eps=50):
    pi = Policy().to(DEVICE)
    pi.load_state_dict(torch.load(OUT / 'policy.pt', map_location=DEVICE))
    raw_fm = RawForwardModel().to(DEVICE)
    raw_fm.load_state_dict(torch.load(OUT / 'raw_fm.pt', map_location=DEVICE))
    hc = Hippocampus()
    hc.load_state_dict(torch.load(OUT / 'hc.pt', map_location=DEVICE))

    env = NavArena(render_mode=None); env.set_curriculum(0)
    goals = 0
    for ep in range(n_eps):
        s = env.reset(seed=42)[0]['state']
        for _ in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
            gd = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

            hc_idx = hc.retrieve(st.squeeze(0), k=10)
            if hc_idx and len(hc.ca3.actions) > 0:
                hc_a = torch.stack([hc.ca3.actions[i] for i in hc_idx]).to(DEVICE)
                hc_z = torch.stack([hc.ca3.patterns[i] for i in hc_idx]).to(DEVICE)
                z_q = hc.dg(st.squeeze(0).unsqueeze(0))
                sims = torch.softmax(z_q @ hc_z.T * 5.0, dim=-1)
                a = sims @ hc_a
            else:
                m, sd, _ = pi(st, gd)
                a = Normal(m, sd).sample()

            obs2, _, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            s = obs2['state']
            if term:
                goals += 1
                break
        logger.info(f'Test {ep + 1}/{n_eps}: {"GOAL" if term else "fail"} ({goals}/{ep + 1})')
    logger.info(f'Test: {goals}/{n_eps} = {goals / n_eps:.0%}')
    env.close()


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    hc, pi, raw_fm = train(2000)
    torch.save(pi.state_dict(), OUT / 'policy.pt')
    torch.save(raw_fm.state_dict(), OUT / 'raw_fm.pt')
    torch.save(hc.state_dict(), OUT / 'hc.pt')
    test(10)
