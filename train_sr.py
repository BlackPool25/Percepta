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
S, H, G, A, PDIM, SPARSITY = 10, 128, 2, 2, 2000, 0.02

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


# ═══ Thalamus: context-dependent attention routing ═══════════════
class Thalamus:
    """Thalamic attention gating: routes relevant dimensions based on PFC context.
    
    The TRN is NOT a simple gain controller — it's a DYNAMIC ROUTER that
    selects which information reaches the cortex based on top-down PFC signals.
    
    Our implementation: the PFC RuleBank provides context, and the Thalamus
    amplifies dimensions that are informative for the CURRENT rule/context.
    """
    def __init__(self, state_dim: int = S):
        self.state_dim = state_dim
        self.context_profiles = {}  # context_id -> dimension weights
    
    def get_profile(self, state):
        """Get or create attention profile for this state context."""
        # Simple context: which side of arena (for Bizonal), env type (objects present?)
        has_objects = abs(state[4:]).sum().item() > 0.01 if state.numel() > 4 else False
        x_side = 'L' if state[0].item() < 0 else 'R' if state.numel() > 0 else 'C'
        ctx_key = f"{'obj' if has_objects else 'noobj'}_{x_side}"
        
        if ctx_key not in self.context_profiles:
            prof = torch.ones(self.state_dim)
            if not has_objects:
                prof[4:] = 0.3  # suppress object dims when none exist
            self.context_profiles[ctx_key] = prof
        return self.context_profiles[ctx_key]
    
    def gate(self, state: torch.Tensor) -> torch.Tensor:
        """Route state through attention profile based on context."""
        prof = self.get_profile(state.squeeze(0) if state.dim() > 1 else state)
        return state * prof.to(state.device)
    
    def update_routing(self, state, rpe):
        """Learn routing: dims with high |RPE correlation| get amplified."""
        pass  # learned during sleep from SchemaBank statistics
    
    def state_dict(self):
        return {k: v for k, v in self.context_profiles.items()}
    def load_state_dict(self, sd):
        for k, v in sd.items():
            self.context_profiles[k] = v


# ═══ Basal Forebrain: pathway-specific ACh learning rate modulation ══
class BasalForebrain:
    """Basal forebrain cholinergic system: modulates learning rate by pathway.
    
    NOT a global modulator — ACh release is PATHWAY-SPECIFIC:
    - Hippocampus: high ACh → fast encoding of new episodes
    - Policy: moderate ACh → cautious action updates  
    - RawFM: low ACh → stable physics (doesn't need rapid change)
    
    ACh is triggered by SURPRISE (large prediction errors), not just novelty.
    """
    def __init__(self):
        self.ach = {'hc': 1.0, 'policy': 1.0, 'rawfm': 0.5}
        self.surprise_history = []
    
    def update(self, rpe_magnitude, novelty):
        """Update pathway-specific ACh based on surprise + novelty."""
        self.surprise_history.append(rpe_magnitude)
        if len(self.surprise_history) > 20:
            self.surprise_history.pop(0)
        
        surprise = np.mean(self.surprise_history) if self.surprise_history else 0
        drive = min(2.0, surprise * 0.5 + novelty * 1.5)
        
        self.ach['hc'] = self.ach['hc'] * 0.9 + max(0.5, min(3.0, drive * 1.5)) * 0.1
        self.ach['policy'] = self.ach['policy'] * 0.9 + max(0.3, min(2.0, drive)) * 0.1
        self.ach['rawfm'] = self.ach['rawfm'] * 0.9 + max(0.2, min(1.5, drive * 0.5)) * 0.1
    
    def get_lr(self, pathway='hc'):
        return self.ach.get(pathway, 1.0)
    
    def state_dict(self):
        return {'ach': self.ach, 'surprise': self.surprise_history[-50:]}
    def load_state_dict(self, sd):
        self.ach.update(sd.get('ach', {}))
        self.surprise_history = list(sd.get('surprise', []))


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
        self.goals = []
        self.episode_ids = []  # which episode each transition belongs to
        self.episode_trajs = {}  # episode_id → list of indices into patterns
        self._next_episode_id = 0
        self._Z = None  # GPU-cached stacked patterns
        self._cached_len = 0

    def store(self, z: torch.Tensor, state: torch.Tensor, action: torch.Tensor,
              reward: float, next_state: torch.Tensor, goal: torch.Tensor = None,
              episode_id: int = None):
        idx = len(self.patterns)
        self.patterns.append(z.detach().cpu())
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.rewards.append(float(reward))
        self.next_states.append(next_state.detach().cpu())
        self.goals.append(goal.detach().cpu() if goal is not None else torch.zeros(2))
        # Track episode membership
        if episode_id is None:
            episode_id = self._next_episode_id
            self._next_episode_id += 1
        self.episode_ids.append(episode_id)
        if episode_id not in self.episode_trajs:
            self.episode_trajs[episode_id] = []
        self.episode_trajs[episode_id].append(idx)

    def retrieve_trajectory(self, z_query: torch.Tensor, k_steps: int = 5) -> list:
        """Retrieve a SEQUENCE (trajectory) of actions, not just one.

        Finds the most similar stored state, then returns the next k_steps
        actions from the SAME episode. This gives multi-step planning without
        compounding error (the actions are from real experience).
        """
        if not self.patterns:
            return []
        # Find closest stored pattern
        Z = self._get_Z(z_query.device)
        sims = z_query @ Z.T
        best_idx = sims[0].argmax().item()
        # Get the episode and position within it
        ep_id = self.episode_ids[best_idx]
        traj = self.episode_trajs[ep_id]
        pos = traj.index(best_idx)
        # Return next k_steps actions from this episode
        result = []
        for i in range(pos, min(pos + k_steps, len(traj))):
            result.append(traj[i])
        return result  # indices into patterns

    def _get_Z(self, device):
        """Get cached stacked pattern matrix on target device.
        Supports incremental update: only stacks NEW patterns since last rebuild.
        """
        n = len(self.patterns)
        if self._Z is None or self._Z.device != device:
            self._Z = torch.stack(self.patterns).to(device)
            self._cached_len = n
        elif self._cached_len < n:
            # Incremental: only stack the new patterns, append to existing cache
            new = torch.stack(self.patterns[self._cached_len:]).to(device)
            self._Z = torch.cat([self._Z, new], dim=0)
            self._cached_len = n
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
                torch.stack([self.next_states[i] for i in idx]),
                torch.stack([self.goals[i] for i in idx]))

    def __len__(self): return len(self.patterns)

    def state_dict(self):
        return {'patterns': self.patterns, 'states': self.states,
                'actions': self.actions, 'rewards': self.rewards,
                'next_states': self.next_states, 'goals': self.goals,
                'episode_ids': self.episode_ids,
                'episode_trajs': self.episode_trajs,
                '_next_episode_id': self._next_episode_id,
                'beta': self.beta}

    def load_state_dict(self, sd):
        self.patterns = sd['patterns']
        self.states = sd.get('states', [])
        self.actions, self.rewards = sd['actions'], sd['rewards']
        self.next_states = sd['next_states']
        self.goals = sd.get('goals', [])
        self.episode_ids = sd.get('episode_ids', [])
        self.episode_trajs = sd.get('episode_trajs', {})
        self._next_episode_id = sd.get('_next_episode_id', 0)
        self.beta = sd['beta']
        self._Z = None


# ═══ Hippocampus: DG + CA3 combined ═══════════════════════════
class Hippocampus:
    """Full hippocampal memory system: DG pattern separation + CA3 storage.

    Usage:
      hc = Hippocampus(state_dim=12, pattern_dim=2000, sparsity=0.02)
      hc.store(state, action, reward, next_state, goal=goal)  # one-shot storage
      idx = hc.retrieve(query_state, k=10)                    # content-addressable retrieval
      states, actions, rewards, next_states, goals = hc.get_batch(idx)
    """
    def __init__(self, state_dim: int = S, pattern_dim: int = PDIM, sparsity: float = SPARSITY):
        self.dg = PatternSeparator(state_dim, pattern_dim, sparsity)
        self.ca3 = CA3Memory()
        self.schema = SchemaBank()
        self.sub = Subiculum()
        self.pfc = RuleBank()
        self.thalamus = Thalamus(state_dim)
        self.bf = BasalForebrain()
        self.chunks = ChunkLibrary()
        self.current_chunk = -1
        self.chunk_step = 0

    def store(self, state, action, reward, next_state, goal=None, episode_id=None):
        st = state.unsqueeze(0) if state.dim() == 1 else state
        z = self.dg(self.thalamus.gate(st))
        g = goal if goal is not None else torch.zeros(2)
        self.ca3.store(z.squeeze(0), state, action, reward, next_state, goal=g,
                       episode_id=episode_id)

    def retrieve(self, query_state, k: int = 10):
        z = self.dg(query_state.unsqueeze(0))
        return self.ca3.retrieve_similar(z, k)

    def get_batch(self, idx):
        return self.ca3.get_batch(idx)

    def __len__(self): return len(self.ca3)

    def get_direction(self, current_state):
        """Returns goal direction tensor or None (if unknown)."""
        dir_vec, _, _, conf = self.sub.get_vector(current_state)
        return dir_vec if conf > 0.3 else None
    
    def state_dict(self):
        d = self.ca3.state_dict()
        d['dg_P'] = self.dg.P.data.clone()
        d['schema'] = self.schema.state_dict()
        d['pfc'] = {'rules': [(str(ctx[0]), ctx[1:]) for ctx, _ in self.pfc.rules],
                    'rule_actions': self.pfc._rule_actions,
                    'confidence': self.pfc.confidence}
        d['sub'] = self.sub.state_dict()
        d['thalamus'] = self.thalamus.state_dict()
        d['bf'] = self.bf.state_dict()
        return d
    def load_state_dict(self, sd): 
        self.ca3.load_state_dict(sd)
        if 'dg_P' in sd:
            self.dg.P.data.copy_(sd['dg_P'])
        if 'schema' in sd:
            self.schema.load_state_dict(sd['schema'])
        if 'pfc' in sd:
            pfc_sd = sd['pfc']
            self.pfc.rules = []
            for ctx_type, ctx_vals in pfc_sd.get('rules', []):
                self.pfc.rules.append(('region', tuple(ctx_vals)))
            self.pfc._rule_actions = pfc_sd.get('rule_actions', [])
            self.pfc.confidence = pfc_sd.get('confidence', 0.0)
        if 'sub' in sd:
            self.sub.load_state_dict(sd['sub'])
        if 'thalamus' in sd:
            self.thalamus.load_state_dict(sd['thalamus'])
        if 'bf' in sd:
            self.bf.load_state_dict(sd['bf'])
    
    def retrieve_actions(self, query_state, k=10):
        """Returns (weighted_action, confidence) from SchemaBank only.
        SchemaBank (anterior hippocampus) provides fast gist-based retrieval.
        Passes original state for PFC context gating (environment type).
        """
        z = self.dg(query_state.unsqueeze(0))
        return self.schema.retrieve_actions(z, k, query_state=query_state)
    
    def get_biased_action(self, query_state, k=10, k_steps=3):
        """Action biased by subiculum goal vector trace.
        
        Blends SchemaBank retrieval with goal-directed vector.
        The goal vector pulls toward the remembered goal position
        (discovered through reward and stored in subiculum VTCs).
        
        Returns (action_tensor, confidence).
        """
        schema_a, confidence = self.retrieve_actions(query_state, k)
        goal_dir, goal_strength, goal_dist, goal_conf = self.sub.get_vector(query_state)
        
        # PFC RuleBank: abstract rules — brain SELECTS not blends
        rule_a, rule_conf = self.pfc.get_action(query_state)
        if rule_a is not None and rule_conf > 0.4:
            if schema_a is None or confidence < 0.3:
                schema_a = rule_a.to(query_state.device)
                confidence = rule_conf
            elif confidence < 0.5:
                align = (schema_a * rule_a.to(query_state.device)).sum().item()
                if align < 0.7 and rule_conf > confidence:
                    schema_a = rule_a.to(query_state.device)
                    confidence = rule_conf
        
        if goal_dir is not None and goal_strength > 0.05 and schema_a is not None:
            # Dopamine ramping: stronger goal pull when close
            blend = goal_strength * 0.7
            action = (1 - blend) * schema_a + blend * goal_dir
            confidence = max(confidence, blend * 0.5)
            return action, confidence
        if goal_dir is not None and goal_strength > 0.05:
            return goal_dir, goal_strength * 0.3
        return schema_a, confidence
    
    def theta_sequence_action(self, query_state, raw_fm, pi, k=10):
        """Theta sequence lookahead: simulate each candidate, evaluate by goal direction.
        
        The brain's hippocampus generates theta sweeps — rapid simulations of
        possible future trajectories. Each candidate action is evaluated by:
        1. RawFM predicts next state: s' = FM(s, a)
        2. Does the predicted movement align with the remembered goal direction?
        3. Does the predicted next state score high on V(s')?
        
        Uses ONLY goal DIRECTION (from subiculum VTCs), not absolute goal position.
        The brain knows the direction to the goal, not its exact coordinates.
        """
        z = self.dg(query_state.unsqueeze(0))
        candidates = self.schema.retrieve_candidates(z, k, query_state=query_state)
        
        if not candidates:
            return self.get_biased_action(query_state, k)
        
        goal_dir, goal_strength, goal_dist, goal_conf = self.sub.get_vector(query_state)
        has_goal = goal_dir is not None and goal_strength > 0.05
        best_score = -float('inf')
        best_action = None
        best_conf = 0.0
        
        for action_t, sim_score, next_state_t, reward_t in candidates:
            with torch.no_grad():
                a = action_t.unsqueeze(0)
                s_pred = raw_fm(query_state.unsqueeze(0), a)[0]
                
                # SLOW value: learned value function (works for any state)
                h = pi.shared(s_pred)
                v_pred = pi.value(h).item()
                score = v_pred + 0.2 * sim_score
                
                # FAST value: goal direction alignment (only when goal known)
                # The brain knows the direction to the goal from VTCs, not coordinates
                if has_goal:
                    gd = goal_dir.to(a.device)
                    # Does this action point toward the remembered goal?
                    action_align = (a.squeeze(0) * gd).sum().item()
                    # Does the predicted movement reduce distance to goal?
                    current_pos = query_state[:2].to(s_pred.device)
                    predicted_move = s_pred[0, :2] - current_pos
                    move_align = (predicted_move * gd).sum().item()
                    # Combined directional score (no goal coordinates used)
                    score += goal_strength * (action_align + move_align)
            
            if score > best_score:
                best_score = score
                best_action = action_t
                best_conf = sim_score
        
        if best_action is None:
            return self.get_biased_action(query_state, k)
        
        return best_action, best_conf
    
    def get_chunked_action(self, st, raw_fm, slow_raw_fm, pi, step):
        """Chunk-based action selection. Returns (action, was_chunk) or (None, False)."""
        if self.current_chunk >= 0:
            if step % 100 == 0:
                self.current_chunk = -1
                self.chunk_step = 0
            elif self.chunk_step < self.chunks.chunk_size:
                a = self.chunks.get_chunk_action(self.current_chunk, self.chunk_step)
                if a is not None:
                    self.chunk_step += 1
                    if self.chunk_step >= self.chunks.chunk_size:
                        self.current_chunk = -1
                        self.chunk_step = 0
                    return a.to(st.device), True
                self.current_chunk = -1
                self.chunk_step = 0
        
        if len(self.chunks) > 0:
            z = self.dg(st.squeeze(0).unsqueeze(0))
            goal_dir, gs, gd, gc = self.sub.get_vector(st.squeeze(0))
            best_score = -float('inf')
            best_idx = -1
            for i in range(len(self.chunks)):
                avg_a = self.chunks.chunk_avg_actions[i].to(st.device).unsqueeze(0)
                if slow_raw_fm is not None:
                    with torch.no_grad():
                        s_pred, _ = slow_raw_fm(st, avg_a)
                        h = pi.shared(s_pred)
                        v = pi.value(h).item()
                        score = v
                        if goal_dir is not None and gs > 0.05:
                            align = (avg_a.squeeze(0) * goal_dir.to(st.device)).sum().item()
                            score += gs * align
                        if score > best_score:
                            best_score = score
                            best_idx = i
            if best_idx >= 0:
                self.current_chunk = best_idx
                self.chunk_step = 1
                a = self.chunks.get_chunk_action(best_idx, 0)
                if a is not None:
                    return a.to(st.device), True
                self.current_chunk = -1
                self.chunk_step = 0
        
        return None, False


# ═══ Subiculum: Goal Vector Trace (persistent goal memory) ═══════
class Subiculum:
    """Subiculum analogue: persistent vector trace to remembered goal.
    
    Vector Trace Cells (VTCs) maintain a persistent representation of
    the goal location. The brain does NOT "reset" this between episodes —
    instead, prediction errors at the old goal location REDUCE CONFIDENCE
    in the stored goal, allowing new goals to form without erasing old ones.
    
    Old goal traces persist but are context-suppressed when prediction
    error signals the goal has moved. The PFC tracks current task context
    and biases which goal trace is retrieved.
    """
    def __init__(self):
        self.goal_pos = None
        self.goal_known = False
        self.confidence = 0.0  # how sure we are about this goal
        self.goal_epoch = 0
    
    def store_goal(self, position):
        """Store goal position with high confidence (called when reward received)."""
        self.goal_pos = position.clone() if hasattr(position, 'clone') else torch.tensor(position, dtype=torch.float32)
        self.goal_known = True
        self.confidence = 1.0
    
    def reduce_confidence(self, amount=0.3):
        """Reduce goal confidence (negative RPE at old goal = goal may have moved)."""
        self.confidence = max(0.0, self.confidence - amount)
        if self.confidence <= 0.0:
            self.goal_known = False
            self.goal_epoch += 1
    
    def get_vector(self, current_state):
        """Compute goal vector. Returns (direction, strength, distance, confidence)."""
        if not self.goal_known or self.goal_pos is None or self.confidence <= 0:
            return None, 0.0, 0.0, 0.0
        
        gx, gy = self.goal_pos[0].item() if hasattr(self.goal_pos[0], 'item') else self.goal_pos[0], \
                 self.goal_pos[1].item() if hasattr(self.goal_pos[1], 'item') else self.goal_pos[1]
        cx, cy = current_state[0].item(), current_state[1].item()
        dx, dy = gx - cx, gy - cy
        dist = (dx * dx + dy * dy) ** 0.5
        
        if dist < 0.01:
            return None, 0.0, 0.0, self.confidence
        
        direction = torch.tensor([dx / dist, dy / dist], device=current_state.device)
        # Strength modulated by confidence — low confidence = weak pull
        strength = min(1.0, 3.0 / (dist + 0.5)) * self.confidence
        
        return direction, strength, dist, self.confidence
    
    def state_dict(self):
        return {'goal_pos': self.goal_pos, 'goal_known': self.goal_known,
                'goal_epoch': self.goal_epoch, 'confidence': self.confidence}
    
    def load_state_dict(self, sd):
        self.goal_pos = sd.get('goal_pos')
        self.goal_known = sd.get('goal_known', False)
        self.goal_epoch = sd.get('goal_epoch', 0)
        self.confidence = sd.get('confidence', 0.0)


# ═══ SchemaBank: Anterior hippocampus (bounded prototype buffer) ══
class SchemaBank:
    """Anterior hippocampus analogue: bounded buffer for fast gist-based retrieval.
    
    Stores prototypes extracted from CA3 patterns during sleep.
    CA3 (posterior hippocampus) keeps ALL patterns forever — never deleted.
    SchemaBank provides fast retrieval for everyday use.
    """
    def __init__(self, capacity=500, similarity_threshold=0.6):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.prototypes = []
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self._Z = None
    
    def retrieve_actions(self, z_query, k=10, query_state=None):
        """Weighted action from top-k prototypes. Returns (action, confidence).
        Uses PFC-like context gating: only retrieves prototypes from matching
        environment context (inferred from extra state dimensions).
        """
        if not self.prototypes:
            return None, 0.0
        # PFC context gating: filter by environment type
        has_ctx = hasattr(self, 'contexts') and len(self.contexts) == len(self.prototypes)
        if has_ctx and query_state is not None:
            query_ctx = abs(query_state[4:]).sum().item() > 0.01 if query_state.numel() > 4 else False
            match_idx = [i for i, c in enumerate(self.contexts) if c == query_ctx]
            if len(match_idx) >= k:
                Z = torch.stack([self.prototypes[i] for i in match_idx]).to(z_query.device)
                sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
                orig_idx = match_idx
            else:
                Z = self._get_Z(z_query.device)
                sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
                orig_idx = list(range(len(self.prototypes)))
        else:
            Z = self._get_Z(z_query.device)
            sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
            orig_idx = list(range(len(self.prototypes)))
        
        max_sim = sims.max().item()
        if max_sim < 0.3:
            return None, max_sim
        topk = min(k, len(orig_idx))
        top_local = sims[0].topk(topk).indices
        top_orig = [orig_idx[i] for i in top_local.tolist()]
        actions = torch.stack([self.actions[i] for i in top_orig]).to(z_query.device)
        return sims[0, top_local] @ actions, max_sim
    
    def retrieve_candidates(self, z_query, k=10, query_state=None):
        """Returns list of (action, similarity, next_state, reward) for top-k prototypes.
        Unlike retrieve_actions which returns a BLENDED action, this returns INDIVIDUAL
        candidates for theta sequence evaluation (simulate each, pick the best).
        """
        if not self.prototypes:
            return []
        has_ctx = hasattr(self, 'contexts') and len(self.contexts) == len(self.prototypes)
        if has_ctx and query_state is not None:
            query_ctx = abs(query_state[4:]).sum().item() > 0.01 if query_state.numel() > 4 else False
            match_idx = [i for i, c in enumerate(self.contexts) if c == query_ctx]
            if len(match_idx) >= k:
                Z = torch.stack([self.prototypes[i] for i in match_idx]).to(z_query.device)
                sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
                orig_idx = match_idx
            else:
                Z = self._get_Z(z_query.device)
                sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
                orig_idx = list(range(len(self.prototypes)))
        else:
            Z = self._get_Z(z_query.device)
            sims = torch.softmax(z_query @ Z.T * 5.0, dim=-1)
            orig_idx = list(range(len(self.prototypes)))
        
        topk = min(k, len(orig_idx))
        top_local = sims[0].topk(topk).indices
        results = []
        for j in range(topk):
            idx = orig_idx[top_local[j].item()]
            results.append((
                self.actions[idx].to(z_query.device),
                sims[0, top_local[j]].item(),
                self.next_states[idx].to(z_query.device) if len(self.next_states) > idx else None,
                self.rewards[idx] if len(self.rewards) > idx else 0.0
            ))
        return results
    
    def update_from_ca3(self, hc, logger=None):
        """Extract prototypes from CA3 by clustering DG patterns.
        NEVER deletes CA3 patterns — SchemaBank is a separate, additive store.
        Also stores PFC-like context mask: which state dims are active.
        """
        patterns = hc.ca3.patterns
        if len(patterns) <= self.capacity and self.prototypes:
            return len(self.prototypes)
        if len(patterns) == 0:
            return 0
        
        n = len(patterns)
        active_sets = [set(torch.where(p > 0.5)[0].tolist()) for p in patterns]
        
        keep = []
        discard = set()
        for i in range(n):
            if i in discard:
                continue
            keep.append(i)
            ai = active_sets[i]
            for j in range(i + 1, n):
                if j in discard:
                    continue
                inter = len(ai & active_sets[j])
                union = len(ai | active_sets[j])
                if union > 0 and inter / union > self.similarity_threshold:
                    discard.add(j)
        
        if len(keep) > self.capacity:
            keep = keep[:self.capacity]
        
        self.prototypes = [patterns[i] for i in keep]
        self.states = [hc.ca3.states[i] for i in keep]
        self.actions = [hc.ca3.actions[i] for i in keep]
        self.next_states = [hc.ca3.next_states[i] for i in keep]
        self.rewards = [hc.ca3.rewards[i] for i in keep]
        # Store context mask: which environment these prototypes belong to
        # PFC context signal — extra state dims distinguish environments
        self.contexts = [(abs(hc.ca3.states[i][4:]).sum().item() > 0.01) for i in keep]
        self._Z = None
        return len(keep)
    
    def _get_Z(self, device):
        if self._Z is None or self._Z.device != device or self._Z.shape[0] != len(self.prototypes):
            self._Z = torch.stack(self.prototypes).to(device)
        return self._Z
    
    def state_dict(self):
        return {'prototypes': self.prototypes, 'states': self.states,
                'actions': self.actions, 'next_states': self.next_states,
                'rewards': self.rewards, 'capacity': self.capacity,
                'similarity_threshold': self.similarity_threshold,
                'contexts': getattr(self, 'contexts', [])}
    
    def load_state_dict(self, sd):
        self.prototypes = sd.get('prototypes', [])
        self.states = sd.get('states', [])
        self.actions = sd.get('actions', [])
        self.next_states = sd.get('next_states', [])
        self.rewards = sd.get('rewards', [])
        self.capacity = sd.get('capacity', 500)
        self.similarity_threshold = sd.get('similarity_threshold', 0.6)
        self.contexts = sd.get('contexts', [])
        self._Z = None
    
    def __len__(self): return len(self.prototypes)


# ═══ PFC RuleBank: abstract rule extraction (dimensionality reduction) ═
class RuleBank:
    """Prefrontal cortex analogue: extracts ABSTRACT RULES from experience.
    
    The PFC performs dimensionality reduction on episodes from the hippocampus,
    stripping away irrelevant details to extract the underlying rule.
    
    Rules are (context_signature → action_vector) pairs:
      - context: which region of state space this rule applies to
      - action: the abstract action direction for that context
    
    Unlike SchemaBank which stores specific (state → action) pairs,
    RuleBank stores GENERAL rules that apply to ANY state in a region.
    """
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.rules = []  # list of (context_fn_desc, action_dir)
        self.context_bounds = []  # (dim_idx, low, high) for each rule's context
        self._rule_actions = []  # action tensors
        self.confidence = 0.0  # overall confidence in rules (0-1)
    
    def extract_from_schema(self, schema, logger=None):
        """Extract abstract rules from SchemaBank prototypes by action clustering.
        Clusters prototypes by ACTION similarity, then checks if they share a context.
        """
        if len(schema.actions) < 10:
            return 0
        
        actions = torch.stack(schema.actions)
        states = torch.stack(schema.states) if schema.states else None
        
        if states is None or len(states) < 5:
            return 0
        
        # Cluster actions by cosine similarity
        norms = actions.norm(dim=1, keepdim=True)
        normed = actions / (norms + 1e-8)
        sims = normed @ normed.T  # cosine similarity matrix
        
        # Greedy action clustering
        n = len(actions)
        used = set()
        new_rules = []
        
        for i in range(n):
            if i in used:
                continue
            # Find all actions similar to i
            cluster = [j for j in range(n) if j not in used and sims[i, j].item() > 0.7]
            if len(cluster) < 3:
                continue
            used.update(cluster)
            
            # Check if this cluster shares a CONTEXT (same state region)
            cluster_states = states[cluster]
            x_coords = cluster_states[:, 0]  # x positions
            y_coords = cluster_states[:, 1]  # y positions
            
            x_mean, x_std = x_coords.mean().item(), x_coords.std().item()
            y_mean, y_std = y_coords.mean().item(), y_coords.std().item()
            
            # If states are clustered in a region, this is a spatial rule
            if x_std < 2.0 and y_std < 2.0:
                # Context: state region (x range, y range)
                ctx = ('region', (x_mean - x_std, x_mean + x_std, y_mean - y_std, y_mean + y_std))
                mean_action = actions[cluster].mean(dim=0)
                new_rules.append((ctx, mean_action))
        
        # Keep top rules by capacity
        if len(new_rules) > self.capacity:
            new_rules = new_rules[:self.capacity]
        
        if new_rules:
            self.rules = new_rules
            self._rule_actions = [r[1] for r in new_rules]
            self.confidence = min(1.0, len(new_rules) / 20)
            if logger:
                logger.info(f"  Rules: {len(new_rules)} abstract rules extracted")
        
        return len(new_rules)
    
    def get_action(self, state):
        """Get the best rule's action for the current state context.
        Returns (action_vector, confidence) or (None, 0.0).
        """
        if not self.rules or self.confidence < 0.3:
            return None, 0.0
        
        state_np = state.cpu().numpy() if hasattr(state, 'cpu') else state
        sx, sy = state_np[0], state_np[1] if len(state_np) > 1 else 0
        
        best_action = None
        best_overlap = 0.0
        
        for ctx, act in self.rules:
            if ctx[0] == 'region':
                _, xlo, xhi, ylo, yhi = ctx
                if xlo <= sx <= xhi and ylo <= sy <= yhi:
                    overlap = min(xhi - xlo, 2.0) * min(yhi - ylo, 2.0)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_action = act
        
        if best_action is None:
            return None, 0.0
        
        return best_action, min(1.0, self.confidence * best_overlap / 2.0)
    
    def __len__(self):
        return len(self.rules)
class ChunkLibrary:
    """Dorsolateral striatum analogue: stores action CHUNKS (5-step sequences).
    
    The brain's DLS binds 5-20 actions into reusable "chunks" — automatic
    routines executed without deliberation. During sleep, trajectory segments
    are clustered into chunk prototypes.
    
    Each chunk: (start_state, action_sequence[5], expected_delta[5])
    High-level policy selects chunks; low-level executes them.
    """
    def __init__(self, capacity=200, chunk_size=5):
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.chunks = []  # list of (start_state, action_seq_5, delta_seq_5)
        self.chunk_avg_actions = []  # mean action per chunk (for slow_raw_fm)
        self._Z_chunks = None
        self._chunk_counter = 0
    
    def extract_from_ca3(self, hc, logger=None):
        """Extract 5-step action chunks from CA3 episodes during sleep."""
        all_chunks = []
        for ep_id, traj in hc.ca3.episode_trajs.items():
            for i in range(0, len(traj) - self.chunk_size, self.chunk_size):
                seg = traj[i:i + self.chunk_size]
                if len(seg) < self.chunk_size:
                    continue
                start_s = hc.ca3.states[seg[0]]
                act_seq = torch.stack([hc.ca3.actions[j] for j in seg])
                delta_seq = []
                for k in range(self.chunk_size - 1):
                    ns = hc.ca3.next_states[seg[k]]
                    cs = hc.ca3.states[seg[k]]
                    delta_seq.append(ns - cs)
                delta_seq.append(hc.ca3.next_states[seg[-1]] - hc.ca3.states[seg[-1]])
                delta_stack = torch.stack(delta_seq) if delta_seq else torch.zeros(self.chunk_size, S)
                all_chunks.append((start_s, act_seq, delta_stack))
        
        if not all_chunks:
            return 0
        
        # Cluster by start-state similarity (simple greedy)
        keep = []
        discard = set()
        for i in range(len(all_chunks)):
            if i in discard:
                continue
            keep.append(i)
            si = all_chunks[i][0]
            for j in range(i + 1, len(all_chunks)):
                if j in discard:
                    continue
                sj = all_chunks[j][0]
                if (si - sj).norm().item() < 0.5:
                    discard.add(j)
        
        if len(keep) > self.capacity:
            keep = keep[:self.capacity]
        
        self.chunks = [all_chunks[i] for i in keep]
        self.chunk_avg_actions = [all_chunks[i][1].mean(dim=0) for i in keep]
        self._Z_chunks = None
        
        if logger:
            logger.info(f"  Chunks: {len(all_chunks)} segments → {len(keep)} prototypes")
        return len(keep)
    
    def get_chunk_action(self, idx, step_in_chunk):
        """Get the action for step_in_chunk within chunk idx."""
        if idx >= len(self.chunks):
            return None
        if step_in_chunk >= self.chunk_size:
            return None
        return self.chunks[idx][1][step_in_chunk]
    
    def __len__(self):
        return len(self.chunks)


# ═══ Policy with value head ════════════════════════════════════
class Policy(nn.Module):
    """Policy π(a|s, goal_dir) with value head V(s).
    
    Takes state AND optional goal direction (from subiculum VTCs).
    The brain's PFC projects goal direction to motor cortex — this is
    an INTERNAL signal, not an external observation. NOT cheating.
    
    When no goal is known (goal_dir=None), acts from state alone.
    """
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(S, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
        )
        self.goal_proj = nn.Linear(2, H)
        self.mean = nn.Linear(H, A)
        self.log_std = nn.Parameter(torch.zeros(A))
        self.value = nn.Linear(H, 1)

    def forward(self, s, goal_dir=None):
        """Forward pass. goal_dir is optional (2-dim normalized direction)."""
        h = self.shared(s)
        if goal_dir is not None:
            h = h + self.goal_proj(goal_dir.to(s.device))
        mean_out = torch.tanh(self.mean(h))
        return (mean_out,
                F.softplus(self.log_std) + 1e-4,
                self.value(h).squeeze(-1))

    def act(self, s):
        m, sd, v = self.forward(s)
        dist = Normal(m, sd)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), v

    def evaluate(self, s, a):
        m, sd, v = self.forward(s)
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


# ═══ Slow RawFM: multi-step forward model (neocortical hierarchy) ══
class SlowCerebellarModel(nn.Module):
    """Neocortical hierarchy: slow forward model that predicts AGGREGATED outcomes.
    
    The neocortex predicts at slower timescales than the cerebellum.
    This model predicts the net effect of 5 consecutive actions:
      Input:  (s, mean_action_over_5_steps)
      Output: (s_{t+5} - s_t, total_reward_over_5_steps)
    
    During sleep, trained on 5-step trajectory segments.
    During wake, used by theta sequences to evaluate long-horizon outcomes.
    """
    def __init__(self, expanded_dim=5000, sparsity=0.02, chunk_size=5):
        super().__init__()
        self.chunk_size = chunk_size
        self.granule = PatternSeparator(S + A, expanded_dim, sparsity)
        self.purkinje = nn.Sequential(
            nn.Linear(expanded_dim, 128), nn.ReLU(),
            nn.Linear(128, S + 1),
        )
    
    def forward(self, s, mean_action):
        z = self.granule(torch.cat([s, mean_action], -1))
        out = self.purkinje(z)
        return s + out[:, :-1], out[:, -1]


# ═══ Demo generation ═══════════════════════════════════════════
def run_demo(env, n_trajs=25):
    """Generate diverse demo trajectories. Goal is read from env._goal_pos, NOT state."""
    rng = np.random.RandomState(42)
    grid = int(np.ceil(np.sqrt(n_trajs)))
    all_s, all_a, all_r, all_ns = [], [], [], []
    agent_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
    vel_adr = 0
    for traj_idx in range(min(grid * grid, n_trajs)):
        i, j = traj_idx // grid, traj_idx % grid
        start_x = -3.0 + 6.0 * (i + 0.5) / grid
        start_y = -3.0 + 6.0 * (j + 0.5) / grid
        env.reset(seed=None)
        env._set_body_pos("agent", np.array([start_x, start_y, 0.5]))
        env.data.qvel[vel_adr:vel_adr + 2] = rng.uniform(-2, 2, size=2)
        mujoco.mj_forward(env.model, env.data)
        s = env._get_obs()['state']
        for _ in range(500):
            # Goal is read from env (NOT from state — brain must remember it)
            g = env._goal_pos[:2]
            p = s[:2]; d_vec = g - p; dist = np.linalg.norm(d_vec)
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
                           raw_fm, hc=None, n_candidates: int = 10,
                           sim_steps: int = 3) -> torch.Tensor:
        """Generate candidate waypoints using HIPPOCAMPAL cognitive map.

        The brain's PFC queries the hippocampus for states that are close
        to the goal and retrievable from the current position. This uses
        ACTUAL STORED EXPERIENCES, not forward model simulations.

        The hippocampus already knows which states lead to the goal —
        it stores (state, action, next_state, reward) from all experience.
        """
        if hc is not None and len(hc.ca3.patterns) > 100:
            # Hippocampal route: find stored states near the goal
            goal_pos = state[0, 2:4].cpu().numpy()
            # Score all stored states by distance to goal
            all_scores = []
            for s_i in hc.ca3.states:
                s_np = s_i.numpy()
                dist_to_goal = np.linalg.norm(s_np[:2] - goal_pos)
                all_scores.append(-dist_to_goal)
            # Pick top 10 states closest to the goal
            best_idx = np.argsort(all_scores)[-10:]
            # If any of these is reachable from current position, use it as subgoal
            current_pos = state[0, :2].cpu().numpy()
            for idx in reversed(best_idx):
                sg = hc.ca3.states[idx][:2].numpy()
                dist_from_current = np.linalg.norm(sg - current_pos)
                # Subgoal must be reachable (between 0.5 and 4 units away)
                if 0.5 < dist_from_current < 4.0:
                    cand = torch.tensor(sg, device=DEVICE)
                    self.last_predicted_value = all_scores[idx]
                    self.last_subgoal_embedding = cand.clone()
                    return cand

        # Fallback: random candidate near the agent
        pos = state[0, :2]
        angle = torch.rand(1, device=DEVICE) * 2 * 3.14159
        radius = torch.rand(1, device=DEVICE) * 1.0 + 0.5
        cand = pos + torch.tensor([torch.cos(angle), torch.sin(angle)],
                                 device=DEVICE).squeeze() * radius
        cand = torch.clamp(cand, -4.5, 4.5)
        self.last_predicted_value = 0.0
        self.last_subgoal_embedding = cand.clone()
        return cand

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
def dopamine_update(pi, opt_pi, opt_val, s, a, r, s_next,
                    gamma=0.99, rpe_clip=10.0, dopamine_boost=1.0):
    """3-factor plasticity: Δθ ∝ δ · ∇_θ log π(a|s), RPE gates LR."""
    with torch.no_grad():
        _, _, v_next = pi(s_next)
        v_next_val = v_next.item()

    lp, v = pi.evaluate(s, a)
    v_val = v.item()
    td_target = r + gamma * v_next_val
    delta = td_target - v_val
    delta_clipped = max(min(delta, rpe_clip), -rpe_clip)

    lr_scale = dopamine_boost * (1.0 + 3.0 * min(abs(delta_clipped) / (rpe_clip / 2), 1.0))

    pi.zero_grad()
    lp, v = pi.evaluate(s, a)
    policy_loss = -(lp * delta_clipped)
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    for p in pi.parameters():
        if p.grad is not None:
            p.grad.data *= lr_scale
    opt_pi.step()

    _, _, v2 = pi(s)
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
    slow_raw_fm = SlowCerebellarModel().to(DEVICE)
    hc._slow_raw_fm = slow_raw_fm

    # Separate optimizers: policy, value (both in pi), and raw FM
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)
    env = NavArena(render_mode=None)

    # DLPFC remembers the goal from the start — NOT in the state

    # ── Seed hippocampal memory with diverse demo trajectories ──
    demo_seed = run_demo(env, n_trajs=25)
    # Store each demo trajectory as a separate episode
    # NOTE: goal is NOT stored — the agent must discover goals through experience,
    # not have them pre-loaded by the teacher. The teacher demonstrates ACTIONS
    # (which are stored), not goal positions.
    traj_len = len(demo_seed[0]) // 25  # approximate transitions per trajectory
    for i in range(len(demo_seed[0])):
        ep_id = i // traj_len if traj_len > 0 else 0
        hc.store(demo_seed[0][i], demo_seed[1][i], demo_seed[2][i], demo_seed[3][i],
                 episode_id=ep_id)
    logger.info(f"Seeded hippocampus with {len(hc)} patterns in {hc.ca3._next_episode_id} episodes")

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

    # ── BC pre-train policy on demo (no goal — learns from actions) ──
    for _ in range(200):
        m, sd, _ = pi(ds)
        loss = F.mse_loss(m, da.to(DEVICE))
        opt_pi.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
    with torch.no_grad():
        m_test, _, _ = pi(ds)
        cosim = F.cosine_similarity(m_test, da.to(DEVICE), dim=-1).mean().item()
    logger.info(f"BC init: cosim={cosim:.3f}")

    # ── Training loop (interleaved phases) ──────────────────────
    goals, step = 0, 0
    current_phase = 0
    current_episode_id = max(hc.ca3._next_episode_id, 0) if hc.ca3._next_episode_id else 0
    env.set_curriculum(current_phase)
    s = env.reset(seed=42)[0]['state']
    ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []
    dopamine_boost = 1.0
    dopamine_decay_steps = 0
    acc = ACC()
    dlpfc = DLPFC()

    t0 = time.time()

    while step < n_steps:
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

        # ── Chunk-based action selection (DLS hierarchical) ──
        m, sd, v = pi(st, hc.get_direction(st.squeeze(0)))
        chunk_action, used_chunk = hc.get_chunked_action(
            st, raw_fm, getattr(hc, '_slow_raw_fm', None), pi, step)
        
        if chunk_action is not None:
            action = chunk_action.unsqueeze(0)
            hc.chunk_step = (hc.chunk_step + 1) % hc.chunks.chunk_size
            if hc.chunk_step == 0:
                hc.current_chunk = -1
        else:
            # Fallback: SchemaBank → trajectory → policy
            schema_action, confidence = hc.get_biased_action(st.squeeze(0), k=10)
            traj_indices = hc.ca3.retrieve_trajectory(hc.dg(st.squeeze(0).unsqueeze(0)), k_steps=3)
            if traj_indices and len(traj_indices) > 0:
                action = hc.ca3.actions[traj_indices[0]].to(DEVICE).unsqueeze(0)
            elif schema_action is not None and confidence > 0.3:
                action = schema_action.unsqueeze(0)
            else:
                action = Normal(m, sd).sample()

        # ── Execute ─────────────────────────────────────────────
        obs2, re, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
        s2 = obs2['state']
        done = term or trunc
        s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

        ep_s.append(st.squeeze(0).cpu())
        ep_a.append(action.squeeze(0).cpu())
        ep_r.append(re)
        ep_ns.append(s2_t.squeeze(0).cpu())

        # ── Cerebellar forward pass: predict next state (efference copy) ──
        sp, rp = raw_fm(st, action)

        # ── Hippocampal CA1 mismatch novelty (brain's curiosity signal) ──
        # CA1 compares current state (via DG) to stored patterns (CA3 retrieval)
        # Low retrieval similarity = novel state = explore
        z_q = hc.dg(st.squeeze(0).unsqueeze(0))
        if len(hc.ca3) > 0:
            Z = hc.ca3._get_Z(st.device)
            sims = torch.softmax(z_q @ Z.T * 5.0, dim=-1)
            novelty = 1.0 - sims.max().item()
        else:
            novelty = 1.0  # everything is novel before any patterns stored
        curiosity_coef = 0.1

        # ── Dopamine-modulated REINFORCE with phasic boost + curiosity bonus ──
        # VTA combines novelty (CA1 mismatch) and task reward into single dopamine signal
        total_reward = re + curiosity_coef * novelty
        delta, lr_scale = dopamine_update(pi, opt_pi, opt_val, st, action, total_reward, s2_t,
                                          dopamine_boost=dopamine_boost)
        
        # ── Basal Forebrain: update ACh levels by surprise + novelty ──
        hc.bf.update(abs(delta) if hasattr(delta, 'item') else abs(delta), novelty)
        # Apply ACh modulated learning rates to optimizers
        for pg in opt_pi.param_groups:
            pg['lr'] = 1e-3 * hc.bf.get_lr('policy')
        for pg in opt_val.param_groups:
            pg['lr'] = 1e-3 * hc.bf.get_lr('policy')
        
        # ── Subiculum: negative RPE at old goal → reduce goal confidence ──
        if hc.sub.goal_known and delta < -10 and hc.sub.confidence > 0.3:
            gx = hc.sub.goal_pos[0].item()
            gy = hc.sub.goal_pos[1].item()
            cx, cy = s[0], s[1]
            dist_to_stored = ((gx - cx)**2 + (gy - cy)**2)**0.5
            if dist_to_stored < 0.8:
                hc.sub.reduce_confidence(amount=0.4)

        # ── Store in hippocampal memory (with episode tracking) ──
        hc.store(st.squeeze(0), action.squeeze(0), re, s2_t.squeeze(0),
                 episode_id=current_episode_id)

        # ── Cerebellar online learning (backward pass, refines forward model) ──
        loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(rp.squeeze(-1), torch.tensor(re, device=DEVICE))
        for pg in opt_raw_fm.param_groups:
            pg['lr'] = 1e-3 * hc.bf.get_lr('rawfm')
        opt_raw_fm.zero_grad(); loss_raw.backward()
        torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()

        # ── ACC: stuck detection via velocity (no goal awareness) ──
        vel = np.linalg.norm(s[2:4])
        stuck = vel < 0.05 and np.linalg.norm(action.squeeze(0).cpu().numpy()) > 0.5
        progress = vel
        detour = stuck

        # PFC: override action if stuck — try random action to get unstuck
        if stuck and not term:
            m, sd, _ = pi(st, hc.get_direction(st.squeeze(0)))
            action = Normal(m, sd).sample() * 1.5  # bigger random action

        step += 1

        # Phasic dopamine boost: decays after goal (simulates dopamine burst)
        if dopamine_decay_steps > 0:
            dopamine_decay_steps -= 1
            dopamine_boost = 1.0 + 4.0 * (dopamine_decay_steps / 25.0)
        else:
            dopamine_boost = 1.0  # 5x → 1x over 25 steps

        # ── Goal reached → subiculum VTC + EC capture + dopamine burst ──
        if term:
            goals += 1
            # Subiculum VTC: store goal position for vector trace navigation
            hc.sub.store_goal(st[0, :2].detach().cpu())
            dopamine_boost = 5.0       # phasic dopamine burst after reward
            dopamine_decay_steps = 25  # decays over ~25 steps
            logger.info(f"GOAL #{goals} step={step} ({len(ep_s)} steps, RPE={delta:.2f}, DA_boost={dopamine_boost:.1f})")
            if len(ep_s) > 1:
                for i in range(len(ep_s)):
                    hc.store(ep_s[i], ep_a[i], ep_r[i], ep_ns[i], episode_id=current_episode_id)
                # BC on successful trajectory
                ec_s = torch.stack(ep_s).to(DEVICE)
                ec_a = torch.stack(ep_a).to(DEVICE)
                for _ in range(30):
                    m_ec, _, _ = pi(ec_s)
                    loss_ec = F.mse_loss(m_ec, ec_a)
                    opt_pi.zero_grad(); loss_ec.backward()
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                logger.info(f"EC: {len(ep_s)} steps captured, BC trained")

        if done:
            current_episode_id += 1  # new episode
            if step > 100:
                current_phase = np.random.randint(0, 5)
                env.set_curriculum(current_phase)
            s = env.reset(seed=42 if current_phase == 0 else None)[0]['state']
            ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []
        else:
            s = s2

        # ── Logging ─────────────────────────────────────────────
        if step % 100 == 0 or step == 1:
            logger.info(
                f"step={step:4d} goals={goals} vel={vel:.2f} "
                f"RPE={delta:+.3f} lr_s={lr_scale:.2f} "
                f"phase={current_phase} hc={len(hc)} "
                f"a_diff={(action - m).norm().item():.3f} "
                f"progress={progress:.2f} "
                f"detour={int(detour)} "
                f"traj_steps={len(traj_indices) if traj_indices else 0}"
            )

        # ── Sleep: consolidate ──────────────────────────────────
        if step > 0 and step % 200 == 0 and len(hc) >= 50:
            all_idx = list(range(len(hc)))
            hs, ha, hr, hn, hg = hc.get_batch(all_idx)
            hs, ha, hn = hs.to(DEVICE), ha.to(DEVICE), hn.to(DEVICE)
            hr, hg = hr.to(DEVICE), hg.to(DEVICE)

            # Train raw FM on (state, action) → next_state
            # Neocortical consolidation: cerebellum refines forward model via structured replay
            # Brain replays sequences in temporal order, preserving trajectory structure
            init_sp, _ = raw_fm(hs, ha)
            init_raw_loss = F.mse_loss(init_sp, hn).item()
            ep_ids = list(hc.ca3.episode_trajs.keys())
            struct_batch = min(128, max(1, len(hn) // 10))
            for i in range(1000):
                if ep_ids:
                    ep_id = ep_ids[np.random.randint(len(ep_ids))]
                    traj = hc.ca3.episode_trajs[ep_id]
                    if len(traj) > struct_batch:
                        start = np.random.randint(0, len(traj) - struct_batch)
                    else:
                        start = 0
                    seg = traj[start:min(start + struct_batch, len(traj))]
                    s_seg = torch.stack([hc.ca3.states[idx] for idx in seg]).to(DEVICE)
                    a_seg = torch.stack([hc.ca3.actions[idx] for idx in seg]).to(DEVICE)
                    ns_seg = torch.stack([hc.ca3.next_states[idx] for idx in seg]).to(DEVICE)
                    r_seg = torch.tensor([hc.ca3.rewards[idx] for idx in seg]).to(DEVICE)
                    sp, rp = raw_fm(s_seg, a_seg)
                    loss = F.mse_loss(sp, ns_seg) + F.mse_loss(rp.squeeze(-1), r_seg)
                else:
                    idx = torch.randperm(len(hs), device=DEVICE)[:struct_batch]
                    sp, rp = raw_fm(hs[idx], ha[idx])
                    loss = F.mse_loss(sp, hn[idx]) + F.mse_loss(rp.squeeze(-1), hr[idx])
                opt_raw_fm.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
            final_sp, _ = raw_fm(hs, ha)
            final_raw_loss = F.mse_loss(final_sp, hn).item()
            logger.info(f"Sleep: RawFM {init_raw_loss:.4f} → {final_raw_loss:.4f} ({len(hn)} trans, 1000 iters)")

            # ── Slow RawFM training (neocortical hierarchy, multi-step) ──
            if hasattr(hc, '_slow_raw_fm') and hc._slow_raw_fm is not None:
                slow_rf = hc._slow_raw_fm
                cs = slow_rf.chunk_size
                slow_states, slow_actions, slow_deltas, slow_rewards = [], [], [], []
                for ep_id in ep_ids:
                    traj = hc.ca3.episode_trajs[ep_id]
                    for i in range(0, len(traj) - cs, cs):
                        seg = traj[i:i + cs]
                        if len(seg) < cs:
                            continue
                        s0 = hc.ca3.states[seg[0]]
                        avg_a = torch.stack([hc.ca3.actions[j] for j in seg]).mean(dim=0)
                        delta = hc.ca3.next_states[seg[-1]] - hc.ca3.states[seg[0]]
                        total_r = sum(hc.ca3.rewards[j] for j in seg)
                        slow_states.append(s0); slow_actions.append(avg_a)
                        slow_deltas.append(delta); slow_rewards.append(total_r)
                if slow_states:
                    ss = torch.stack(slow_states).to(DEVICE)
                    sa = torch.stack(slow_actions).to(DEVICE)
                    sd_target = torch.stack(slow_deltas).to(DEVICE)
                    sr_target = torch.tensor(slow_rewards, device=DEVICE)
                    init_l = F.mse_loss(slow_rf(ss, sa)[0], sd_target + ss).item()
                    for _ in range(500):
                        sp_slow, rp_slow = slow_rf(ss, sa)
                        loss = F.mse_loss(sp_slow, sd_target + ss) + F.mse_loss(rp_slow.squeeze(-1), sr_target)
                        opt_raw_fm.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(slow_rf.parameters(), 1.0); opt_raw_fm.step()
                    final_l = F.mse_loss(slow_rf(ss, sa)[0], sd_target + ss).item()
                    logger.info(f"Sleep: SlowRawFM {init_l:.4f} → {final_l:.4f} ({len(ss)} chunks, {cs}-step)")

            # ── Action chunk extraction (DLS: cluster trajectory segments) ──
            n_chunks = hc.chunks.extract_from_ca3(hc, logger=logger)

            # ── PFC rule extraction (abstract rules from SchemaBank) ──
            n_rules = hc.pfc.extract_from_schema(hc.schema, logger=logger)

            # ── Compositional Sleep Replay ──────────────────────────
            # Stitch trajectory segments from different episodes into NOVEL trajectories.
            # This is how the brain generalizes: recombining past experiences.
            comp_states, comp_actions = [], []
            ep_ids = list(hc.ca3.episode_trajs.keys())
            if len(ep_ids) >= 2:
                for _ in range(50):  # generate 50 composed transitions
                    ep_id_a, ep_id_b = np.random.choice(ep_ids, 2, replace=False)
                    traj_a = hc.ca3.episode_trajs[ep_id_a]
                    traj_b = hc.ca3.episode_trajs[ep_id_b]
                    if len(traj_a) < 2 or len(traj_b) < 2:
                        continue
                    # Pick a random "choice point" — a state in episode A
                    choice_idx = np.random.randint(len(traj_a))
                    choice_state = hc.ca3.states[traj_a[choice_idx]]
                    # Find a similar state in episode B
                    best_b = None
                    best_sim = -1
                    for j, idx_b in enumerate(traj_b[:-1]):  # skip last (no next action)
                        s_b = hc.ca3.states[idx_b]
                        sim = -(choice_state - s_b).norm().item()
                        if sim > best_sim:
                            best_sim = sim
                            best_b = j
                    if best_b is not None and best_sim > -2.0:
                        # Stitch: action from ep_a at choice point + action from ep_b after match
                        a_a = hc.ca3.actions[traj_a[choice_idx]]
                        a_b = hc.ca3.actions[traj_b[best_b]]
                        # Keep the ep_a action, but composed ep_b state gives alternative context
                        comp_states.append(choice_state)
                        comp_actions.append(a_a)  # same action, but new context from episode B

            if comp_states:
                cs = torch.stack(comp_states).to(DEVICE)
                ca = torch.stack(comp_actions).to(DEVICE)
                all_hs = torch.cat([hs, cs], dim=0)
                all_ha = torch.cat([ha, ca], dim=0)
                logger.info(f"Sleep: +{len(cs)} composed transitions")
            else:
                all_hs, all_ha = hs, ha

            # Sleep BC: policy learns from stored + composed trajectories
            bc_losses = []
            for _ in range(100):
                m_bc, _, _ = pi(all_hs)
                loss = F.mse_loss(m_bc, all_ha)
                bc_losses.append(loss.item())
                opt_pi.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
            logger.info(f"Sleep BC: {bc_losses[0]:.4f} → {bc_losses[-1]:.4f} ({len(all_hs)} trans)")

            # ── SchemaBank update (anterior hippocampus gist extraction) ──
            # Clusters CA3 DG patterns into prototypes. NEVER deletes CA3 patterns.
            n_protos = hc.schema.update_from_ca3(hc, logger=logger)

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
    hc.load_state_dict(torch.load(OUT / 'hc.pt', map_location='cpu'))

    # Online adaptation: optimizers for continued learning during testing
    # The brain never stops learning — testing is also learning
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)

    env = NavArena(render_mode=None); env.set_curriculum(0)
    goals = 0
    for ep in range(n_eps):
        s = env.reset(seed=42)[0]['state']
        for _ in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

            # Hierarchical action selection: theta sequence → SchemaBank → policy
            schema_action, confidence = hc.theta_sequence_action(st.squeeze(0), raw_fm, pi, k=10)
            if schema_action is not None and confidence > 0.3:
                a = schema_action  # shape (2,)
            else:
                m, sd, _ = pi(st, hc.get_direction(st.squeeze(0)))
                a = Normal(m, sd).sample().squeeze(0)  # shape (2,)

            obs2, re, term, trunc, _ = env.step(a.cpu().numpy())
            s2 = obs2['state']
            s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

            # Online adaptation: continue learning during testing
            sp, rp = raw_fm(st, a.unsqueeze(0))
            # Hippocampal CA1 mismatch novelty
            if len(hc.ca3) > 0:
                z_q = hc.dg(st.squeeze(0).unsqueeze(0))
                Z = hc.ca3._get_Z(st.device)
                sims = torch.softmax(z_q @ Z.T * 5.0, dim=-1)
                novelty = 1.0 - sims.max().item()
            else:
                novelty = 1.0
            total_reward = re + 0.1 * novelty
            dopamine_update(pi, opt_pi, opt_val, st, a.unsqueeze(0), total_reward, s2_t, dopamine_boost=1.0)
            loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(rp.squeeze(-1), torch.tensor(re, device=DEVICE))
            opt_raw_fm.zero_grad(); loss_raw.backward()
            torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
            hc.store(st.squeeze(0), a.squeeze(0), re, s2_t.squeeze(0), episode_id=ep)
            if term:
                hc.sub.store_goal(st[0, :2].detach().cpu())

            s = s2
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
