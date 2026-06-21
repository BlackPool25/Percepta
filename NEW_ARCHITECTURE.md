# Percepta — New Architecture Reference (v3.0)

**Version:** 3.0 (Post-Research Revision)  
**Date:** June 21, 2026  
**Purpose:** Complete reference for the redesigned brain-inspired architecture after extensive 2025-2026 research and 50+ experimental iterations.

---

## 1. Project Vision

Build an AI agent that learns continuously from experience in a simulated world — without catastrophic forgetting, without pre-built backbones, and without being a benchmark-optimization project. The agent should:

- Learn from few examples via one-shot Hebbian binding
- Consolidate important experiences slowly into stable long-term knowledge
- Revise beliefs when contradicted by new evidence
- Retrieve and recombine past experiences via content-addressable memory
- Do all this without transformers, without frozen pretrained weights, and without unbounded memory growth

**Current best result:** 76 goals during training, 100% test on fixed-start navigation after 2000 training steps. 25 demo trajectories (5×5 grid). 28 seconds training.

---

## 2. What We Tried and What Failed

### Failed Approaches (from v1.0-v2.0)

| Approach | What Was Tried | Why It Failed |
|----------|---------------|---------------|
| **Discrete QMemory as primary storage** | Store (state, action, Q) as 2000 list entries | Not scalable. 2000 discrete entries vs brain's distributed weights. Can't form concepts. |
| **Continuous blending for BG gate** | `h_out = g * h_new + (1-g) * h` — smooth blend of old/new | Brain uses BINARY gate (Go/NoGo). Continuous blend has no neuroscience basis. |
| **RPE directly updates value weights** | `w += α · δ · φ(s)` — dopamine RPE directly trains w | In brain, RPE trains the BG GATE, not the value function. Value is learned separately via TD. |
| **Training forward model on exploration data** | Sleep consolidation trained on 70% exploration + 30% demo | 76:1 exploration-to-demo ratio caused catastrophic forgetting (demo loss: 0.02 → 256). |
| **PPO on purely negative rewards** | Standard PPO with distance-based reward (−0.1 × dist) | Policy collapses to "do nothing" because all actions lead to negative rewards. No positive signal. |
| **Full buffer clear after sleep** | Clear all transitions after consolidation | Demo knowledge lost. Forward model divergence. Brain doesn't clear — it interleaves. |
| **Sequential curriculum phases** | Phase 0 (fixed) → Phase 1 (random start) → Phase 2 (random goal) | Catastrophic forgetting between phases. Brain uses interleaved training, not sequential. |
| **Attention-weighted action retrieval** | `action = Σ(attention × stored_actions)` — soft blend of all stored actions | Blends demo actions with exploration noise. Dilutes goal direction from 0.7 to ~0.1. |
| **Using φ(s) as forward model input** | `fm(φ(s), a) → φ(s')` — forward model in φ-space | φ space drifts as SRNet/W updates during training. Forward model predictions become stale (demo_loss=267). |
| **φ-space policy** | `π(φ(s), goal)` — policy takes successor features as input | φ drifts during training, policy target keeps changing. Can't stabilize learning. |
| **Q(s) = φ(s)^T · w value function** | Linear value function in φ-space | w collapses negative from sparse positive rewards. All states get negative Q. |
| **Diagonal Fisher metaplasticity** | Fisher importance → per-parameter LR | Diagonal Fisher has Spearman ρ=0.34 for dense layers (project's own finding). Structurally insufficient. |
| **GRU policy** | GRU for temporal context | Navigation state has full observability (position + velocity). GRU adds complexity without benefit. Makes BC training harder (needs correct hidden state sequences). |
| **Frozen teacher distillation** | Teacher forward model + distillation loss | Only needed for φ-space FM (which drifts). Raw-state FM doesn't drift. |

### Approaches That Worked (v3.0 — Current Architecture)

| Approach | Success | Evidence |
|----------|---------|----------|
| **Hippocampal episodic control** | **Critical** | DG pattern separation + CA3 content-addressable retrieval replaces PPO/latent planning. Achieved 76 goals, 100% test. |
| **DG pattern separation (k-WTA)** | **Critical** | 2000-dim sparse codes (2% active). Fixed random projection prevents interference. Enables one-shot Hebbian storage. |
| **CA3 similarity-weighted action retrieval** | **Critical** | Weighted average of actions from 10 most similar past states. `softmax(β · z_q @ Z.T) @ actions`. More robust than hard top-K. |
| **Dopamine-modulated REINFORCE** | **Critical** | RPE-gated policy LR: LR = base × (1 + 3·|δ|/5). Separate optimizers for policy and value. Clipped RPE at 10. |
| **Phasic dopamine boost** | Key | 5x LR burst after goal reach, decaying over 25 steps. Simulates midbrain dopamine burst firing. |
| **Raw-state forward model** | Key | Trained continuously on every (s,a)→s' transition. No φ-space drift. Predicts raw coordinates for cerebellar planning. |
| **EC capture** | Key | Successful trajectories stored permanently in hippocampus. Immediate BC training on success. |
| **Grid-based demo coverage** | Key | 5×5 grid over [-3,3]² state space ensures diverse demo data. Fixed seed for reproducibility. |
| **GPU-cached pattern stacking** | Moderate | Lazily-built cached Z matrix avoids O(N²) per-step stacking. 2000 patterns → ~10ms rebuild only on miss. |
| **Sleep BC consolidation** | Moderate | Policy BC-trained on ALL hippocampal patterns every 200 steps. Loss drops from ~0.8 to ~0.01 over training. |

---

## 3. Non-Negotiable Requirements

| # | Requirement | Brain Mechanism | Current Status | Implementation |
|---|---|---|---|---|
| 1 | **Fast-write episodic memory** | Hippocampus (DG+CA3) | ✅ **IMPLEMENTED** | DG: fixed random projection + k-WTA (2% sparsity). CA3: one-shot storage + attention retrieval. |
| 2 | **Content-addressable retrieval** | CA3 autoassociative | ✅ **IMPLEMENTED** | `softmax(β · z_q @ Z.T) @ actions`. GPU-cached Z matrix for O(1) retrieval. |
| 3 | **Sparse representations** | DG pattern separation | ✅ **IMPLEMENTED** | k-WTA at 2% sparsity on 2000-dim sparse codes. Fixed random projection. |
| 4 | **Dopamine-modulated plasticity** | Striatum | ✅ **IMPLEMENTED** | RPE gates policy LR: 1-4x based on |δ|. Phasic 5x boost for 25 steps after reward. |
| 5 | **Separate policy/value systems** | Go/NoGo pathways | ✅ **IMPLEMENTED** | Separate optimizers with gradient isolation. Value: stable TD. Policy: RPE-gated REINFORCE. |
| 6 | **Continuous cerebellar learning** | Cerebellum | ✅ **IMPLEMENTED** | Raw-state FM trained on every (s,a)→s' transition. No batching needed. |
| 7 | **Slow weight consolidation** | Neocortex | ✅ **PARTIAL** | Sleep BC on hippocampal patterns. Missing: per-synapse importance for selective consolidation. |
| 8 | **Online non-IID training** | Continuous streaming | ✅ **IMPLEMENTED** | Single-pass streaming. No epochs. Hippocampus stores continuously. Sleep consolidates. |

---

## 4. Current Architecture (train_sr.py) — v3.0

### Components

```
HippocampalMemory (DG + CA3):
  - DG: PatternSeparator (S=12 → 2000, 2% k-WTA, fixed random projection)
  - CA3: pattern list + GPU-cached Z matrix + associated (action, reward, next_state)
  - store(z, state, action, reward, next_state): one-shot Hebbian storage
  - retrieve(query_state, k=10): return indices of 10 most similar stored patterns
  - GPU cache: _Z lazily built on first retrieval, invalidated on store

Policy (Motor Cortex + OFC):
  - Shared MLP: [s(12-dim), goal_dir(2-dim)] → 128 → ReLU → 128 → ReLU → h
  - Mean head: h → tanh → action(2-dim)
  - log_std: learnable parameter (initialized 0)
  - Value head: h → V(s) (scalar) — trained via TD, SEPARATE optimizer

RawForwardModel (Cerebellum):
  - [s(12-dim), a(2-dim)] → 128 → ReLU → 128 → [s'(12-dim), r(1-dim)]
  - Trained continuously: every step, MSE(s_pred, s_actual) + MSE(r_pred, r_actual)
  - No φ-space involvement — no drift issues

DopamineUpdate (Striatum):
  - δ = r + γV(s') - V(s)    (RPE, clipped to ±10)
  - Policy: Δθ ∝ δ · ∇_θ log π(a|s)    (REINFORCE with RPE gate)
  - Value: Δθ ∝ ∇_θ (V(s) - TD_target)²    (TD learning, separate opt)
  - LR_eff = dopamine_boost × (1 + 3·|δ|/5)    (RPE gates learning rate)
  - dopamine_boost: 5.0 after reward, decays to 1.0 over 25 steps
```

### Data Flow

```
Wake (every step):
  1. Observe s (12-dim raw state), compute goal direction gd
  2. Hippocampal retrieval: z = DG(s), find 10 nearest patterns in CA3
  3. Action = softmax(β · z_q @ Z_nearest.T) @ actions_nearest
  4. Execute action → observe s', r
  5. Compute RPE δ = r + γ·V(s') - V(s)
  6. Policy update: opt_pi on -(log π(a|s) · δ_clipped)  × LR_eff
  7. Value update: opt_val on MSE(V(s), r + γ·V(s'))
  8. Store (s, a, r, s') in hippocampal memory (DG → sparse code → CA3)
  9. Cerebellar update: train RawFM on (s, a) → (s', r)

  10. If goal reached (r > 0):
      - Trigger phasic dopamine boost (5x for 25 steps)
      - Store entire episode trajectory in hippocampus (EC capture)
      - BC train policy on successful trajectory (30 iterations)

Sleep (every 200 steps):
  1. Train RawFM on ALL hippocampal patterns: (state, action) → (next_state, reward)
  2. BC train policy on ALL hippocampal patterns: (state, goal) → action
  3. Update per-synapse importance (for future metaplasticity)
```

---

## 5. What Changed from v2.0

### Removed Components

| Component | Reason | Replacement |
|-----------|--------|-------------|
| **SRNet (φ-space)** | φ drifts during training, breaks all downstream components | **REMOVED entirely.** Raw state used instead. |
| **Q(s) = φ(s)^T · w** | w collapses negative from sparse rewards | **REMOVED.** Raw-state value function V(s) via TD. |
| **φ-space ForwardModel** | φ drift makes predictions stale (d_loss > 200) | **REPLACED** by raw-state ForwardModel (no drift). |
| **TransitionBuffer** | Flat list, no content-addressability | **REPLACED** by HippocampalMemory (DG+CA3). |
| **PPO (actor)** | Policy collapses with all-negative rewards | **REPLACED** by dopamine-modulated REINFORCE. |
| **PPO (critic)** | Shared optimizer interferes with policy | **REPLACED** by separate value optimizer (TD learning). |
| **Latent planning (Q prediction)** | Requires working forward model + Q function | **REPLACED** by hippocampal episodic retrieval. |
| **Distillation (teacher FM)** | Only needed for φ-space drift | **REMOVED.** Raw FM doesn't drift. |
| **GRU policy** | Unnecessary for fully-observable MDP | **REPLACED** by MLP policy (simpler, faster). |
| **Fisher importance** | Diagonal Fisher ρ=0.34 for dense layers | **REMOVED.** Pending better importance metric. |
| **Demo actions as planning candidates** | Depends on forward model + Q | **REMOVED.** Hippocampus stores all successful actions. |

### Added Components

| Component | Purpose | Key Innovation |
|-----------|---------|----------------|
| **HippocampalMemory (DG+CA3)** | Episodic storage + content-addressable retrieval | Pattern separation prevents interference. GPU cache for speed. |
| **Dopamine-modulated REINFORCE** | 3-factor plasticity for policy learning | RPE gates both direction AND magnitude. Separate policy/value optimizers. |
| **Phasic dopamine boost** | Enhanced plasticity after reward | 5x LR for 25 steps after goal. Simulates midbrain burst firing. |
| **Grid-based demo coverage** | Diverse initial demonstration data | 5×5 grid over [-3,3]² ensures coverage. Fixed seed for reproducibility. |
| **Raw-state ForwardModel** | Predict s' from (s, a) in raw coordinates | Trained continuously on every transition. No drift. |
| **Cerebellar online learning** | Continuous prediction error minimization | Every step: (s, a) → s'. No batching needed. |
| **GPU-cached Z matrix** | Fast similarity search | Lazy-built, invalidated on store. O(1) retrieval amortized. |

---

## 6. Current Performance Metrics

| Test Scenario | Success Rate | Notes |
|---|---|---|
| Fixed start → Fixed goal (Phase 0) | **100%** (10/10) | 76 goals during 2000-step training. 28s training time. |
| Random start → Fixed goal | Not tested | Requires hippocampal generalization |
| Random start → Random goal | Not tested | Requires hippocampal generalization |
| Random maze | Not tested | Requires forward model on raw state |

---

## 7. Architectural Decisions Log (v3.0)

### D14: Hippocampal memory replaces TransitionBuffer (2026-06-21)
**Decision:** TransitionBuffer (flat list) replaced by HippocampalMemory (DG+CA3).
**Why:** The brain doesn't store transitions in a flat list. It uses pattern separation (DG) to convert similar inputs to different sparse codes, and CA3 for content-addressable retrieval. This enables one-shot storage of millions of patterns without interference.
**Neuroscience mapping:** DG performs pattern separation via fixed mossy fiber connections (k-WTA). CA3 performs autoassociative completion via recurrent collaterals.

### D15: Episodic control replaces latent planning (2026-06-21)
**Decision:** Action selection via hippocampal retrieval, not forward model simulation.
**Why:** The latent planning required both an accurate forward model AND a reliable Q-function. Both proved unreliable (φ-space drift + w collapse). Episodic control retrieves actions directly from similar past experiences, bypassing both issues.
**Neuroscience mapping:** The hippocampus retrieves entire episodes during decision-making. The PFC doesn't simulate every candidate — it recalls what worked before in similar situations.

### D16: Dopamine-modulated REINFORCE replaces PPO (2026-06-21)
**Decision:** 3-factor plasticity with RPE-gated LR replaces PPO surrogate loss.
**Why:** PPO with all-negative rewards collapses the policy to "do nothing." REINFORCE with RPE (δ) as the third factor gives positive updates for good outcomes and negative updates for bad ones. The RPE-gated LR provides adaptive step sizes based on surprise.
**Neuroscience mapping:** Striatal plasticity follows a 3-factor rule: pre × post × dopamine (RPE). PPO's surrogate loss has no biological basis.

### D17: Raw-state policy replaces φ-space (2026-06-21)
**Decision:** Policy takes raw 12-dim state + goal, not φ(s).
**Why:** φ(s) drifts during training as SRNet weights change. The policy's input distribution keeps shifting, preventing stable learning. Raw state doesn't drift.
**Neuroscience mapping:** The striatum receives direct sensory input from cortex, not just abstract successor features.

### D18: Separate optimizers for policy and value (2026-06-21)
**Decision:** Two separate Adam optimizers — one for policy parameters, one for value head.
**Why:** Shared optimizer caused gradient interference. The large policy gradients (from RPE-gated LR) destabilized the value function, and vice versa. Separate optimizers allow each to have its own learning rate and gradient statistics.
**Neuroscience mapping:** The brain has separate neuromodulatory systems for learning: dopamine (striatum/policy) vs. acetylcholine (cortex/value). These operate at different timescales.

### D19: GPU-cached Z matrix (2026-06-21)
**Decision:** Cache the stacked pattern matrix on GPU, rebuild lazily on invalidate.
**Why:** `torch.stack(self.patterns).to(device)` on every retrieval creates O(N·D) overhead per step. With 4000+ 2000-dim patterns, this is 32 MB per call. Caching reduces to O(1) for the common case.
**Trade-off:** Invalidation on store means the cache rebuilds once per step during training. But in the test (no storage), it builds once and reuses for 25000 retrievals.

---

## 8. Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `train_sr.py` | Main training + testing | Hippocampus, CA3Memory, PatternSeparator, Policy, RawForwardModel |
| `env_nav.py` | Custom MuJoCo arena | NavArena (walls, objects, goal) |
| `hopfield_memory.py` | Legacy DG + CA3 tests | PatternSeparator, ModernHopfieldMemory (retained for reference) |
| `test_generalization.py` | 4-phase generalization test | run_phase, metrics tracking |

---

## 9. Open Questions

1. **How does the hippocampus generalize to novel states?** The current system retrieves the 10 most similar patterns from storage. If a novel state doesn't match any stored pattern closely enough, the weighted average may produce poor actions. Solutions: (a) the policy can generalize via BC training on diverse stored experiences, (b) the raw FM can simulate outcomes for novel states.

2. **Should we reintroduce a forward model for planning in novel states?** The raw FM is trained continuously. For states where the hippocampus has no close match, the FM could simulate the best candidate action. This is the brain's dual-system: hippocampus for known situations, cerebellum for novel ones.

3. **How to handle random start/goal/maze?** The 5×5 grid demo covers the state space uniformly. For random start, the hippocampus may not have stored a pattern close to the start position. Solutions: (a) more diverse demo data, (b) policy must generalize via BC, (c) online adaptation during test.

4. **When does hippocampal memory need consolidation into policy weights?** Currently, sleep BC trains the policy on ALL hippocampal patterns. The policy can then act without hippocampal retrieval. But the hippocampus is still needed for novel states. Over time, the policy should internalize common patterns and only rely on hippocampus for outliers.

5. **Should we add vector RPE (heterogeneous dopamine)?** Different state dimensions may need different learning rates. For example, position is more important than object positions. Vector RPE would provide dimension-specific dopamine signals.

---

## 10. Testing Methodology

### Quick Test (1 min)
```bash
cd /home/lightdesk/Downloads/Projects/Percepta
.venv/bin/python3 train_sr.py  # 2000 steps training + 10 episode test
```

### Generalization Test (10 min)
```bash
.venv/bin/python3 test_generalization.py  # 4 phases, 50 episodes each
```

### Metrics Tracked
- Goals reached (training and test)
- RPE values (should spike positive at goal, negative elsewhere)
- Hippocampal pattern count
- Sleep BC loss (should decrease over training)
- Policy action magnitude (should stay in [-1, 1])
- Hippocampal retrieval similarity (should be > 0.5 for good matches)

---

## 11. Entry Point for Next Agent

### Read First
1. `NEW_ARCHITECTURE.md` — This document (architecture, decisions, current state)
2. `train_sr.py` — Current implementation (the working system)
3. `env_nav.py` — Custom MuJoCo environment

### Run First
```bash
cd /home/lightdesk/Downloads/Projects/Percepta
.venv/bin/python3 train_sr.py  # 2000 steps training + 10 episode test
```

### Build Priority (Next)
1. **Vector RPE (heterogeneous dopamine)** — Different learning rates for position vs. velocity vs. object dimensions. Medium priority.
2. **Compositional replay during sleep** — Recombine parts of different episodes to generate novel successful actions. Medium priority.
3. **Forward model for novel-state planning** — Use raw FM to simulate when hippocampus has no close match. Low priority.
4. **Per-synapse metaplasticity (activation-based)** — Use the existing `importance` buffer for selective weight consolidation during sleep. Low priority.
5. **Random start/goal generalization** — Train with curriculum phases starting from Phase 1, using hippocampal retrieval for action selection. Low priority.

### Key Files
| File | What to Change |
|------|----------------|
| `train_sr.py` | Add vector RPE, compositional replay, metaplastic consolidation. |
| `test_generalization.py` | Rewrite to use new architecture. Add 4-phase test. |
| `env_nav.py` | Already supports curriculum phases — no changes needed. |

### Git History (last 5 commits)
```
5838ba3 Phasic dopamine boost + grid demo coverage. Train: 76 goals. Test: 100%.
02dda41 Working hippocampal architecture + dopamine REINFORCE.
3bb12d5 Per-synapse metaplasticity + raw-state FM + logging.
29fff3e Distillation + demo candidates. Train: 16 goals. Test: 48%.
048d4fd Forward planning + top-K retrieval. Train: 76 goals. Test: 100%.
```

---

## 12. Next Build Priority

1. **Vector RPE (heterogeneous)** — Midbrain DA. Different dopamine signals for different state dimensions. Not global scalar. High impact for generalization.
2. **Compositional replay during sleep** — Hippocampus. Generate NOVEL action sequences by recombining known primitives, not just replay. Medium impact.
3. **Generalization to random start/goal** — Test and potentially fix the hippocampal generalization gap. Requires policy to act without close hippocampal matches.
4. **Per-synapse metaplasticity** — Add selective weight consolidation using the existing activation-based importance buffer during sleep BC.
5. **Cerebellar planning for novel states** — When hippocampus retrieves low-similarity patterns, fall back to raw FM simulation for action selection.
