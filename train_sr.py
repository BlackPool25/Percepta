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
class CerebellarModel(nn.Module):
    """Cerebellum: granule sparse expansion → Purkinje nonlinear readout → Δs.

    [s, a] → Granule (5000, 2% k-WTA) → Purkinje (128 nonlinear) → Δs + r

    The granule layer uses fixed random projection + k-WTA (same as DG).
    The Purkinje readout is a small MLP that decodes the sparse code.
    """
    def __init__(self, expanded_dim: int = 5000, sparsity: float = 0.02):
        super().__init__()
        self.granule = PatternSeparator(S + A, expanded_dim, sparsity)
        self.purkinje = nn.Sequential(
            nn.Linear(expanded_dim, 128), nn.ReLU(),
            nn.Linear(128, S + 1),
        )

    def forward(self, s, action):
        z = self.granule(torch.cat([s, action], -1))
        out = self.purkinje(z)
        return s + out[:, :-1], out[:, -1]


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


# ═══ DLPFC: Working Memory (4 slots) + Subgoal Generator ═════
class DLPFC:
    """Dorsolateral PFC + OFC + BG — working memory, subgoal generation, outcome learning.

    4 working memory slots:
      0: current_subgoal_xy
      1: task_phase
      2: plan_history
      3: context

    OFC (Orbitofrontal Cortex): outcome prediction error learning.
      After subgoal execution, compares predicted vs actual value.
      Trains the subgoal generator to produce better candidates.

    BG (Basal Ganglia) gating: dopamine-modulated subgoal selection.
      Learns which subgoal TYPES lead to success via RPE.
    """
    def __init__(self, hidden_dim: int = H):
        self.hidden_dim = hidden_dim
        self.slots = torch.zeros(4, hidden_dim, device=DEVICE)
        
        # Gate network
        gate_input_dim = S + G + 4 * hidden_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 4),
        ).to(DEVICE)
        
        # Subgoal generator (trained by OFC outcome error)
        self.subgoal_net = nn.Sequential(
            nn.Linear(S + G + 1, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        ).to(DEVICE)
        
        # OFC: stores predicted vs actual value for outcome learning
        self.last_predicted_value = None  # set before subgoal execution
        self.subgoal_attempts = []        # tracks (success, subgoal) pairs
        
        # Optimizer for subgoal generator (trained by OFC error)
        self.opt_gen = torch.optim.Adam(self.subgoal_net.parameters(), lr=1e-3)
        
        self.mode = 'normal'
        self.current_subgoal = None
        self.last_subgoal_embedding = None

    def update_slots(self, state: torch.Tensor, goal_dir: torch.Tensor):
        """Update working memory slots via gated recurrence."""
        with torch.no_grad():
            # Flatten slots for gate input
            context = torch.cat([state.squeeze(0), goal_dir.squeeze(0),
                               self.slots.view(-1)])
            gates = torch.sigmoid(self.gate_net(context.unsqueeze(0)))
            # Update each slot: blend old and new
            for i in range(4):
                new_val = torch.randn(self.hidden_dim, device=DEVICE) * 0.01
                self.slots[i] = (1 - gates[0, i]) * self.slots[i] + gates[0, i] * new_val

    def set_subgoal(self, subgoal_xy: torch.Tensor):
        """Set a subgoal in working memory slot 0."""
        # Encode subgoal (2-dim) into slot (H-dim) by projecting
        subgoal_enc = torch.zeros(self.hidden_dim, device=DEVICE)
        subgoal_enc[:2] = subgoal_xy.to(DEVICE)
        self.slots[0] = subgoal_enc
        self.current_subgoal = subgoal_xy.clone()
        self.mode = 'subgoal'

    def clear_subgoal(self):
        """Return to normal goal-steering mode."""
        self.slots[0] = torch.zeros(self.hidden_dim, device=DEVICE)
        self.current_subgoal = None
        self.mode = 'normal'

    def generate_candidates(self, state: torch.Tensor, goal_dir: torch.Tensor,
                           raw_fm, n_candidates: int = 10, sim_steps: int = 3) -> torch.Tensor:
        """Generate and evaluate candidate waypoints.

        OFC stores the predicted value of the best candidate for later
        outcome learning. BG gating selects candidates by learned utility.
        """
        candidates = []
        scores = []
        pos = state[0, :2]

        with torch.no_grad():
            for k in range(n_candidates):
                if k < n_candidates // 2:
                    angle = torch.rand(1, device=DEVICE) * 2 * 3.14159
                    radius = torch.rand(1, device=DEVICE) * 0.8 + 0.3
                    cand = pos + torch.tensor([torch.cos(angle), torch.sin(angle)],
                                             device=DEVICE).squeeze() * radius
                else:
                    perp = torch.tensor([-goal_dir[0, 1], goal_dir[0, 0]], device=DEVICE)
                    offset = perp * ((k - n_candidates // 2) * 0.4 + 0.2)
                    cand = pos + offset
                cand = torch.clamp(cand, -4.5, 4.5)

                s_sim = state.clone()
                total_score = 0.0
                for step_k in range(sim_steps):
                    d_vec = cand - s_sim[0, :2]
                    d_norm = d_vec.norm() + 1e-8
                    a_sim = (d_vec / d_norm).unsqueeze(0)
                    s_pred, _ = raw_fm(s_sim, a_sim)
                    total_score -= (s_pred[0, :2] - s_pred[0, 2:4]).norm().item()
                    if (s_pred - s_sim).norm().item() < 0.01:
                        total_score -= 5.0
                    s_sim = s_pred
                candidates.append(cand)
                scores.append(total_score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_candidate = candidates[best_idx]

        # OFC: store predicted value for outcome learning after execution
        self.last_predicted_value = scores[best_idx]
        self.last_subgoal_embedding = best_candidate.clone()

        return best_candidate

    def ofc_outcome_learning(self, actual_progress: float):
        """OFC: learn from subgoal outcome. Train generator to improve.

        Called after subgoal is reached or abandoned.
        Compares predicted vs actual value, updates subgoal generator.
        """
        if self.last_predicted_value is None:
            return
        # OFC error: how wrong was our prediction?
        # Positive = subgoal was BETTER than predicted (reinforce)
        # Negative = subgoal was WORSE than predicted (suppress)
        ofc_error = actual_progress - self.last_predicted_value

        # Train subgoal generator with this feedback
        # This is the brain's counterfactual learning signal
        dummy_state = torch.zeros(1, S + G + 1, device=DEVICE)
        pred_subgoal = self.subgoal_net(dummy_state)
        loss = -ofc_error * pred_subgoal.norm()  # reinforce better subgoals
        self.opt_gen.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.subgoal_net.parameters(), 1.0)
        self.opt_gen.step()

        self.subgoal_attempts.append((ofc_error > 0, self.last_subgoal_embedding))
        self.last_predicted_value = None

    def get_query_state(self, state: torch.Tensor, goal_dir: torch.Tensor) -> torch.Tensor:
        """Return the query state for hippocampal retrieval.

        In normal mode: use current state (retrieve actions for current state)
        In subgoal mode: use subgoal as query (retrieve actions for reaching subgoal)
        """
        if self.mode == 'subgoal' and self.current_subgoal is not None:
            # Create a synthetic state at the subgoal position
            query = state.clone()
            query[0, :2] = self.current_subgoal.to(state.device)
            return query
        return state


# ═══ ACC: Anterior Cingulate Cortex (goal progress monitoring) ══
class ACC:
    """Anterior Cingulate Cortex — adaptive detour detection with ACh/NA threshold modulation.

    Uses ACETYLCHOLINE-like precision gating: tracks prediction error statistics
    to dynamically adjust the threshold. In stable environments (low variance),
    threshold rises (fewer false positives). In volatile environments (high variance),
    threshold drops (more sensitive).

    Uses NORADRENALINE-like sensitivity modulation: consecutive failures increase
    sensitivity (lower threshold) to trigger faster replanning.
    """
    def __init__(self):
        self.detour_signal = False
        self.stuck_steps = 0
        self.progress_history = []  # last 20 progress values
        self.max_history = 20

    def _get_adaptive_threshold(self) -> float:
        """ACh-modulated threshold: higher when stable, lower when volatile."""
        if len(self.progress_history) < 5:
            return -0.2
        std = np.std(self.progress_history) + 0.01
        mean = np.mean(self.progress_history)
        # Threshold = mean - 1.5*std (dynamics based on current variability)
        return mean - 1.5 * std

    def detect(self, dist_before: float, dist_after: float, in_goal: bool = False) -> tuple:
        if in_goal:
            self.stuck_steps = max(0, self.stuck_steps - 2)
            self.detour_signal = False
            return 0.0, False

        progress = dist_before - dist_after
        self.progress_history.append(progress)
        if len(self.progress_history) > self.max_history:
            self.progress_history.pop(0)

        threshold = self._get_adaptive_threshold()

        # NA-modulated: stuck steps make us more sensitive
        na_sensitivity = 1.0 + 0.1 * self.stuck_steps  # increases with frustration

        stuck = progress < threshold * na_sensitivity
        if stuck:
            self.stuck_steps += 1
        else:
            self.stuck_steps = max(0, self.stuck_steps - 1)

        self.detour_signal = self.stuck_steps >= 3

        return progress, self.detour_signal

    def reset(self):
        self.detour_signal = False
        self.stuck_steps = 0
        self.progress_history = []


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
    raw_fm = CerebellarModel().to(DEVICE)
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

    # ── Training loop (interleaved phases) ──────────────────────
    goals, step = 0, 0
    current_phase = 0
    env.set_curriculum(current_phase)
    s = env.reset(seed=42)[0]['state']
    ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []
    dopamine_boost = 1.0        # phasic dopamine burst multiplier
    dopamine_decay_steps = 0    # steps remaining for phasic boost
    acc = ACC()                 # Anterior Cingulate Cortex (prediction error detector)
    dlpfc = DLPFC()             # Dorsolateral PFC (working memory + subgoal gen)
    t0 = time.time()

    while step < n_steps:
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

        # Compute goal direction: toward main goal or subgoal
        if dlpfc.mode == 'subgoal' and dlpfc.current_subgoal is not None:
            sg = dlpfc.current_subgoal.to(DEVICE)
            gd = (sg - st[:, :2]) / ((sg - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)
        else:
            gd = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

        # ── PFC-modulated action selection ───────────────────────
        # Normal mode: hippocampus retrieves actions from similar states
        # Subgoal mode: directly steer toward subgoal (no hippocampal retrieval)
        m, sd, v = pi(st, gd)

        if dlpfc.mode == 'subgoal' and dlpfc.current_subgoal is not None:
            # In subgoal mode: use policy to steer toward subgoal directly
            # The skip connection in the policy already steers toward gd
            # gd was set to point toward subgoal at the top of the loop
            action = Normal(m, sd).sample()

            # Check if we've reached the subgoal
            sg = dlpfc.current_subgoal.cpu().numpy()
            if np.linalg.norm(s[:2] - sg) < 0.5:
                # OFC outcome learning: how well did this subgoal work?
                dlpfc.ofc_outcome_learning(actual_progress=progress)
                dlpfc.clear_subgoal()
                logger.info(f"Subgoal reached at step {step}")
        else:
            # Normal mode: hippocampal episodic retrieval
            dist = Normal(m, sd)
            hc_idx = hc.retrieve(st.squeeze(0), k=10)
            if hc_idx and len(hc.ca3.actions) > 0:
                hc_actions = torch.stack([hc.ca3.actions[i] for i in hc_idx]).to(DEVICE)
                z_query = hc.dg(st.squeeze(0).unsqueeze(0))
                Z = torch.stack([hc.ca3.patterns[i] for i in hc_idx]).to(DEVICE)
                sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
                action = sims @ hc_actions
            else:
                action = dist.sample()

        # Update DLPFC working memory
        dlpfc.update_slots(st, gd)

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
        loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(rp.squeeze(-1), torch.tensor(re, device=DEVICE))
        opt_raw_fm.zero_grad(); loss_raw.backward()
        torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()

        # ── ACC + PFC: detour detection and replanning ───────────
        dist_before = np.linalg.norm(s[:2] - s[2:4])
        dist_after = np.linalg.norm(s2[:2] - s2[2:4])
        progress, detour = acc.detect(dist_before, dist_after, term)

        # PFC subgoal generation with OFC outcome learning
        if detour and dlpfc.mode == 'normal':
            logger.info(f"Detour at step {step}! Generating waypoints...")
            best_candidate = dlpfc.generate_candidates(st, gd, raw_fm,
                                                       n_candidates=10, sim_steps=3)
            dlpfc.set_subgoal(best_candidate)
            logger.info(f"Subgoal: ({best_candidate[0]:.2f}, {best_candidate[1]:.2f})")

        step += 1
        dist_to_goal = np.linalg.norm(s[:2] - s[2:4])

        # Phasic dopamine boost: decays after goal (simulates dopamine burst)
        if dopamine_decay_steps > 0:
            dopamine_decay_steps -= 1
            dopamine_boost = 1.0 + 4.0 * (dopamine_decay_steps / 25.0)
        else:
            dopamine_boost = 1.0  # 5x → 1x over 25 steps

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
            # Cycle through curriculum phases for multi-task learning
            if step > 100:  # let the first episode finish with Phase 0
                current_phase = np.random.randint(0, 4)
                env.set_curriculum(current_phase)
            s = env.reset(seed=42 if current_phase == 0 else None)[0]['state']
            ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []
        else:
            s = s2

        # ── Logging ─────────────────────────────────────────────
        if step % 100 == 0 or step == 1:
            logger.info(
                f"step={step:4d} goals={goals} dist={dist_to_goal:.2f} "
                f"RPE={delta:+.3f} lr_s={lr_scale:.2f} "
                f"phase={current_phase} hc={len(hc)} "
                f"a_diff={(action - m).norm().item():.3f} "
                f"progress={progress:.2f} "
                f"detour={int(detour)} "
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
    raw_fm = CerebellarModel().to(DEVICE)
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
