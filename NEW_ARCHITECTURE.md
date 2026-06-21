# Percepta — New Architecture Reference (v2.0)

**Version:** 2.0 (Post-Research Revision)  
**Date:** June 21, 2026  
**Purpose:** Complete reference for the redesigned brain-inspired architecture after extensive 2025-2026 research and 50+ experimental iterations.

---

## 1. Project Vision

Build an AI agent that learns continuously from experience in a simulated world — without catastrophic forgetting, without pre-built backbones, and without being a benchmark-optimization project. The agent should:

- Learn from few examples via one-shot Hebbian binding
- Consolidate important experiences slowly into stable long-term knowledge
- Revise beliefs when contradicted by new evidence
- Use prospective planning (simulate before acting) rather than trial-and-error
- Do all this without transformers, without frozen pretrained weights, and without unbounded memory growth

**Current best result:** 48% test success on fixed-start navigation after 2000 training steps with 26-step demo. 16 goals during training.

---

## 2. What We Tried and What Failed

### Failed Approaches

| Approach | What Was Tried | Why It Failed |
|----------|---------------|---------------|
| **Discrete QMemory as primary storage** | Store (state, action, Q) as 2000 list entries | Not scalable. 2000 discrete entries vs brain's distributed weights. Can't form concepts. |
| **Continuous blending for BG gate** | `h_out = g * h_new + (1-g) * h` — smooth blend of old/new | Brain uses BINARY gate (Go/NoGo). Continuous blend has no neuroscience basis. |
| **RPE directly updates value weights** | `w += α · δ · φ(s)` — dopamine RPE directly trains w | In brain, RPE trains the BG GATE, not the value function. Value is learned separately via TD. |
| **Training forward model on exploration data** | Sleep consolidation trained on 70% exploration + 30% demo | 76:1 exploration-to-demo ratio caused catastrophic forgetting (demo loss: 0.02 → 256) |
| **PPO on purely negative rewards** | Standard PPO with distance-based reward (−0.1 × dist) | Policy collapses to "do nothing" because all actions lead to negative rewards. No positive signal. |
| **Full buffer clear after sleep** | Clear all transitions after consolidation | Demo knowledge lost. Forward model divergence. Brain doesn't clear — it interleaves. |
| **Sequential curriculum phases** | Phase 0 (fixed) → Phase 1 (random start) → Phase 2 (random goal) | Catastrophic forgetting between phases. Brain uses interleaved training, not sequential. |
| **Attention-weighted action retrieval** | `action = Σ(attention × stored_actions)` — soft blend of all stored actions | Blends demo actions with exploration noise. Dilutes goal direction from 0.7 to ~0.1. |
| **Using φ(s) as forward model input** | `fm(φ(s), a) → φ(s')` — forward model in φ-space | φ space drifts as SRNet/W updates during training. Forward model predictions become stale (demo_loss=267). |

### Approaches That Worked

| Approach | Success | Evidence |
|----------|---------|----------|
| **RBF similarity (not softmax) for confidence** | Key | Softmax dilutes with pattern count. RBF depends only on nearest neighbor distance. |
| **Top-K action retrieval (not attention blend)** | Key | Pure demo actions avoid blending noise. Achieved 100% test in early experiments. |
| **Forward model re-trained on CURRENT φ values** | Key | Every sleep cycle, re-compute φ for stored states with current SRNet. Adapts to drift. |
| **Demo actions as candidates in latent planning** | Key | Forward model simulates each candidate, picks highest predicted Q. Achieved 48% test. |
| **Latent planning (simulate before acting)** | Key | Hippocampal theta sweep analogue. 10x improvement over act-then-learn. |
| **Distillation (teacher forward model)** | Moderate | Student preserves physics knowledge while adapting. Prevents catastrophic forgetting. |
| **|δ|-gated buffer retention** | Moderate | Surprising experiences (high |δ|) retained, expected ones evicted. Prevents noise accumulation. |
| **Never fully clear the buffer** | Critical | Demo transitions permanent. Only low-|δ| exploration evicted. Prevents knowledge loss. |
| **Behavioral cloning on demo first** | Critical | Pre-trains policy to match demo actions. Without this, policy produces random actions. |

---

## 3. Non-Negotiable Requirements

| # | Requirement | Brain Mechanism | Current Status | Implementation |
|---|---|---|---|---|
| 1 | Fast-write + slow-write split | Hippocampus + Neocortex | ✅ DESIGNED | TransitionBuffer (temporary) + SRNet weights (permanent). Buffer never fully cleared. |
| 2 | Sparse representations | DG pattern separation | ✅ DESIGNED | PatternSeparator: k-WTA at 2% sparsity on position. Fixed random projection. |
| 3 | Modular routing | Anatomical specialization | ❌ NOT STARTED | Deferred until interference is demonstrated. |
| 4 | Replay mechanism | SWR consolidation | ✅ IMPLEMENTED | Interleaved replay every 50 steps. Demo retained permanently. High-|δ| prioritized. |
| 5 | Importance-weighted updates | STC hypothesis | ✅ PARTIAL | |δ|-gated retention. Demo importance=1.0. Missing: per-synapse Fisher importance for effective LR. |
| 6 | Neuromodulation | Dopamine, NE, ACh | ✅ PARTIAL | RPE computed per transition. Gates buffer retention. Missing: dopamine modulates POLICY learning rate, not just buffer. |
| 7 | Content-addressable retrieval | CA3 autoassociative | ✅ DESIGNED | Hopfield attention on DG-separated position keys. RBF similarity for confidence. |
| 8 | Online non-IID training | Continuous streaming | ✅ IMPLEMENTED | Single-pass streaming. No epochs. Buffer fills online. PPO updates every 64 steps. |

---

## 4. Current Architecture (train_sr.py)

### Components

```
SRNet (Neocortex):
  - φ(s) = net(s): 12-dim state → 256-dim successor features (distributed representation)
  - w: 256-dim reward weight vector
  - Q(s) = φ(s)^T · w: parametric value
  - Fisher importance tracking (per-parameter gradient history)

Policy (PFC + Motor Cortex):
  - GRU([φ(s), goal_dir]) → hidden state (working memory)
  - mean/std → action distribution
  - Value head: V(s) for PPO critic

ForwardModel (Cerebellum):
  - [φ(s), a] → predicted φ(s') + predicted reward
  - Frozen teacher copy for distillation
  - Re-trained on CURRENT φ values every 200 steps

TransitionBuffer (Hippocampus):
  - Raw (s, a, r, s') transitions with |δ| tracking
  - Demo: PERMANENT (importance=1.0, never evicted)
  - Exploration: retained if high |δ|, evicted if low |δ|
  - NEVER fully cleared

Latent Planning (Hippocampal theta sweep):
  - Sample 10 candidate actions from policy + all demo actions
  - For each: simulate φ', r' through forward model
  - Compute Q(s') = r' + 0.99 · φ'^T · w
  - Pick action with highest predicted Q
  - Executes before acting (simulate-then-act, not act-then-learn)
```

### Data Flow

```
Wake (every step):
  1. Observe s → compute φ(s) = SRNet(s)
  2. Policy: π(φ(s), goal_dir) → action distribution
  3. Latent planning: sample candidates, simulate through fm, pick best Q
  4. Execute best action → observe s', r
  5. Compute RPE δ = r + 0.99·Q(s') - Q(s)
  6. Store (s, a, r, s') in buffer with |δ|
  7. TD update on w: w += 1e-4 · δ · φ(s)
  8. PPO update every 64 steps (actor + critic)

Sleep (every 200 steps):
  1. Evaluate forward model on demo + exploration data
  2. Re-train forward model on CURRENT φ values of demo states
  3. Sync teacher ← student after re-training
  4. Evict low-|δ| exploration transitions
```

---

## 5. What's Still Missing (Critical for 100% Generalization)

| Missing Component | Brain Region | Function | Priority |
|---|---|---|---|
| **Per-synapse metaplasticity** | Entire cortex | Fisher importance ⇒ per-parameter LR. Important weights learn slowly. | HIGH |
| **Dopamine-modulated policy LR** | Striatum | RPE gates policy learning rate. High |δ| = learn faster, low |δ| = protect. | HIGH |
| **Vector RPE (heterogeneous)** | Midbrain DA | Different dopamine signals for different state dimensions. Not global scalar. | MEDIUM |
| **Compositional replay during sleep** | Hippocampus | Generate NOVEL action sequences by recombining known primitives, not just replay. | MEDIUM |
| **Hierarchical action chunks** | BG-thalamic loops | Group primitive actions into reusable "skills" (subgoal→primitive→subgoal). | LOW |
| **Cerebellar forward model for RAW state** | Cerebellum | Predict s' from (s, a) directly (12-dim), not in φ-space. Avoids φ drift issues. | LOW |
| **EC on successful exploration** | Hippocampus | When exploration accidentally achieves goal, store that trajectory with high Q. | HIGH |

### Priority Justification

1. **Per-synapse metaplasticity**: The policy collapses because all weights learn at the same rate. Demo-important weights should have LR → 0 while novel weights remain plastic. This is the single highest-impact fix.

2. **Dopamine-modulated policy LR**: Currently, RPE only gates buffer retention. It should also gate the policy's learning rate. After a successful goal reach (high +δ), the policy should immediately lock in the good actions.

3. **EC on successful exploration**: When the agent accidentally reaches the goal (which it does 16 times in 2000 steps), the successful trajectory should be stored with high importance. Currently, these successes are stored but not prioritized differently from unsuccessful exploration.

---

## 6. Current Performance Metrics

| Test Scenario | Success Rate | Notes |
|---|---|---|
| Fixed start → Fixed goal (Phase 0) | 48% (24/50) | Demo actions available in latent planning |
| Random start → Fixed goal | Not tested | Requires trained policy |
| Random start → Random goal | Not tested | Requires metaplasticity |
| Random maze | Not tested | Requires forward model on raw state |

---

## 7. Architectural Decisions Log (Post-Research)

### D9: Memory stores RAW TRANSITIONS, not Q-values (2026-06-21)
**Decision:** TransitionBuffer stores (s, a, r, s') tuples with |δ| tracking. No Q-values stored.
**Why:** Q-values are computed on-the-fly by SRNet. Stored Q-values become stale when w is updated. Raw transitions are timeless — they can be replayed with the current w.
**Neuroscience mapping:** Hippocampus stores raw episodic experiences, not computed values. Values are computed in PFC/OFC at retrieval time.
**Previous mistake:** QMemory stored Q-values which diverged when w was updated via TD.

### D10: Forward model uses CURRENT φ values (2026-06-21)
**Decision:** Every 200 steps, re-compute φ for stored demo states with the current SRNet, then re-train the forward model on these updated φ values.
**Why:** SRNet's φ space drifts during training (w updates change φ). If the forward model is trained on stale φ values, its predictions become invalid (demo_loss=267).
**Neuroscience mapping:** The cerebellum receives current sensory context through mossy fibers. It doesn't use stale representations.

### D11: Distillation for forward model protection (2026-06-21)
**Decision:** Frozen teacher + trainable student forward model. Distillation loss = ||student - teacher||² + 0.1·||student - actual||².
**Why:** Prevents catastrophic forgetting of physics knowledge while allowing adaptation to φ drift.
**Neuroscience mapping:** The cerebellum has a slow-learning internal model that preserves core dynamics while allowing rapid adaptation to body changes.

### D12: Latent planning overrides policy (2026-06-21)
**Decision:** Before every action, simulate 10+ candidate actions through the forward model and pick the one with highest predicted Q.
**Why:** The policy collapses under PPO with all-negative rewards. Latent planning with demo action candidates bypasses the broken policy and provides good actions.
**Neuroscience mapping:** Hippocampal theta sweeps simulate future trajectories before action execution. CA1 evaluates each candidate and selects the best.

### D13: Buffer NEVER fully cleared (2026-06-21)
**Decision:** Demo transitions are permanent (importance=1.0). Only low-|δ| exploration transitions are evicted. Buffer is never fully cleared.
**Why:** The brain doesn't erase everything after sleep. Important experiences are retained indefinitely. Full buffer clearing caused catastrophic forgetting in both the forward model and policy.
**Neuroscience mapping:** Hippocampal memories that survive multiple sleep cycles are consolidated. Demo = consolidated, exploration = unconsolidated.

---

## 8. Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `train_sr.py` | Main training + testing | SRNet, Policy, ForwardModel, TransitionBuffer |
| `env_nav.py` | Custom MuJoCo arena | NavArena (walls, objects, goal) |
| `hopfield_memory.py` | DG + CA3 memory | PatternSeparator, ModernHopfieldMemory |
| `test_generalization.py` | 4-phase generalization test | run_phase, metrics tracking |
| `step9_embodied_learning.py` | Earlier episodic control attempts | ImportanceWeightedMemory, AugmentedPolicyHead |
| `train_curriculum.py` | Sequential curriculum training | Curriculum phases |
| `test_curriculum.py` | Curriculum test from pretrained | Phase progression |

---

## 9. Open Questions

1. **Why does the policy collapse under PPO despite positive Q values from latent planning?** The latent planning selects good actions, but PPO trains the policy on the SELECTED action, not on the policy's own action. The policy never learns to generate good actions independently.

2. **Should we separate the policy's training from PPO entirely?** Use behavioral cloning on successful buffer trajectories instead of PPO on negative rewards. The buffer stores successful episodes — extract them and train the policy via supervised learning.

3. **Is φ-space the right representation for the forward model?** The φ space drifts during training, requiring constant re-training. Would raw state (12-dim) be more stable? Trade-off: φ captures abstract features, raw state is pixel-level.

4. **How to achieve 100% test on random start/goal?** Requires the policy to generate good actions without demo candidates. This requires either: (a) metaplasticity to prevent policy collapse, (b) BC on diverse successful trajectories, or (c) model-based planning (Dreamer-style).

5. **At what point does the hippocampal buffer need consolidation into neocortical weights?** Currently the buffer is permanent (2000 entries). But the brain consolidates hippocampus → neocortex and clears the hippocampal trace. When should we clear the buffer and rely entirely on SRNet weights?

---

## 10. Testing Methodology

### Quick Test (5 min)
```bash
python3 train_sr.py  # 2000 steps training + 50 episode test
```

### Generalization Test (10 min)
```bash
python3 test_generalization.py  # 4 phases, 50 episodes each
```

### Metrics Tracked
- Goals reached (training and test)
- Q values (should be positive when near goal)
- Forward model demo_loss vs exploration_loss
- Policy mean magnitude (should stay above 0.5 if not collapsed)
- Buffer composition (demo vs high-|δ| vs low-|δ|)
- Latent planning action difference (should be non-zero when planning changes action)

---

## 11. Entry Point for Next Agent

### Read First
1. `NEW_ARCHITECTURE.md` — This document (architecture, decisions, failures, priorities)
2. `train_sr.py` — Current implementation (the working system)
3. `env_nav.py` — Custom MuJoCo environment

### Run First
```bash
cd /home/lightdesk/Downloads/Projects/Percepta
.venv/bin/python3 train_sr.py  # 2000 steps training + 50 episode test
```

### Build Priority (from Section 12)
1. **Per-synapse metaplasticity** — Fisher importance for per-parameter policy LR. Prevents policy collapse.
2. **Dopamine-modulated policy LR** — High RPE events trigger high policy update LR.
3. **EC on successful exploration** — Store successful trajectories with high importance.
4. **Forward model on raw state** — Avoid φ-space drift issues by predicting s' from (s, a).

### Key Files
| File | What to Change |
|------|----------------|
| `train_sr.py` | Main architecture. Add metaplasticity, dopamine-gated LR, buffer BC training. |
| `env_nav.py` | Environment. Add curriculum phases, random maze walls, random goals. |
| `test_generalization.py` | 4-phase test. Enable memory TD updates during test for fast adaptation. |

### Git History (last 10 commits)
```
29fff3e Distillation + demo candidates. Train: 16 goals. Test: 48%.
048d4fd Forward planning + top-K retrieval. Train: 76 goals. Test: 100% (fixed start).
7379ded PFC-query gating + trajectory replay. Demo-only: 32%.
59091ba NEC-style Q-memory + binary BG gate + RPE trains gate.
566b037 SR architecture: allocentric memory, TD on w, RBF blending.
```

---

## 12. Next Build Priority

1. **Per-synapse metaplasticity** — Fisher importance → per-parameter LR for policy. Prevents policy collapse.
2. **Dopamine-modulated policy LR** — High |δ| events (successful goal reach) trigger high policy update LR.
3. **EC on successful exploration** — Store successful exploration trajectories with high importance.
4. **Forward model on raw state** — Add a second forward model that predicts s' from (s, a) in 12-dim space (avoids φ drift).
5. **BC on successful buffer trajectories** — Extract successful episodes from buffer, train policy via supervised learning.
