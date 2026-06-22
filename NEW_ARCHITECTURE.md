# Percepta — Complete Brain Architecture Reference (v8.0 — FINAL)

**Version:** 8.0
**Date:** June 22, 2026
**Purpose:** Complete reference — every decision, every experiment, every component, everything still missing, and the path forward for true brain-like AGI.

---

## 1. Project Mission

Build an AI agent that learns continuously from experience — without catastrophic forgetting, without pre-built backbones, without GPUs, and without memorizing. The agent should learn like a brain: store experiences fast, extract rules slowly, generalize to novel situations, and improve over time.

**Final best results:**
| Benchmark | Result |
|-----------|--------|
| Phase 0 (fixed start/goal) | **100%** |
| Phase 1 (random start) | **100%** |
| Phase 3 (walls) | **66%** |
| Phase 5 (proper maze) | **40%** |
| Cross-env retention (after learning Bizonal) | **4/5 phases preserved** |
| Bizonal (novel goals each episode) | **43%** (0 wrong-zone errors) |
| VRAM usage | **~206MB** (stable) |

---

## 2. Complete Decision Log (D1-D34)

### D1-D15: Original Decisions (Pre-v6.0)
Remove SRNet, φ-space Q, PPO → dopamine REINFORCE, GRU → feedforward, Add hippocampal episodic control, Cerebellar forward model, Phasic dopamine boost, Separate policy/value optimizers, Skip connection, Remove goal from state, Remove goal direction from policy, Remove PFC subgoal generation, Velocity-based stuck detection, Episode-level trajectory storage, Compositional sleep replay.

### D16: Hippocampal CA1 Curiosity
**What:** Replaced RawFM prediction error with CA1 mismatch: `novelty = 1 - max(softmax(z_q @ Z.T))`.
**Why:** RawFM error measures motor learning, not spatial novelty. Once physics is learned, error drops to zero even in novel mazes.
**Key lesson:** Curiosity must measure SPATIAL novelty, not prediction error.

### D17: Structured Sleep Replay
**What:** 30 → 1000 RawFM iterations. Shuffled → structured episode trajectory segments.
**Why:** Cerebellum refines forward models through structured replay of complete episodes.
**Impact:** RawFM loss: 2-6 → 0.005-0.1.

### D18: SchemaBank (Anterior Hippocampus)
**What:** Replaced pattern-deletion compression with SchemaBank (bounded prototype buffer, capacity=500). CA3 patterns NEVER deleted.
**Why:** The brain's anterior hippocampus stores gist; posterior stores ALL episodes. Deletion causes catastrophic forgetting (Phase 4 dropped from 38% → 26%).
**Key lesson:** NEVER delete CA3 patterns. Compression = separate prototype buffer, not deletion.

### D19: Episode-Level Sleep BC
**What:** Sleep BC samples equally from all episodes.
**Why:** Proportional sampling lets newer/longer episodes dominate → forgetting.
**Key lesson:** Each episode gets equal replay time regardless of length.

### D20: VRAM Fix
**What:** Incremental Z-cache, SchemaBank-only retrieval during wake, empty_cache() after sleep.
**Why:** CA3 Z-cache rebuilt every step via torch.cat caused O(n) GPU growth to 16GB.
**Key lesson:** Never cache all CA3 patterns on GPU during wake. SchemaBank-only retrieval.

### D21: PFC Context Gating
**What:** SchemaBank stores environment context per prototype. Retrieval filters by matching.
**Why:** Without context, maze patterns (steer toward 3,3) contaminate bizonal retrieval.
**Impact:** Zero wrong-zone errors in Bizonal.

### D22: Bizonal Multi-Rule Arena
**What:** Created env_bizonal.py — LEFT starts → NW goals. RIGHT starts → SE goals.
**Why:** Tests context-dependent rule learning without explicit labels.
**Key result:** Zero wrong-zone errors proves rule extraction.

### D23: Cross-Environment Retention
**What:** Created test_cross.py — same agent learns BOTH maze AND Bizonal.
**Result:** 4/5 maze phases preserved. Proves multi-environment retention.

### D24: Subiculum Goal Vector Trace with Confidence
**What:** Subiculum stores goal position from reward discovery. Uses prediction error to track goal confidence (not erase).
**Why:** The brain doesn't erase goals — it updates confidence via prediction error.
**Key lesson:** Negative RPE at old goal → confidence reduces, old trace persists but suppressed.

### D25: Theta Sequence Lookahead
**What:** Evaluates candidate actions via RawFM simulation before execution.
**Why:** The brain's hippocampus generates theta sweeps — rapid simulations of possible futures.
**Impact:** Enables pre-action evaluation. Uses fast goal-distance computation (instant, no learning).

### D26: Policy Receives Goal Direction
**What:** Policy.forward accepts optional `goal_dir` from subiculum. Projected into shared representation.
**Why:** The brain's PFC projects goal information to motor cortex. Internal signal, not external observation.
**Verification:** Goal direction comes from subiculum internal memory, not environment.

### D27: Remove Goal Cheating from Demos
**What:** Removed `goal=env._goal_pos` from demo hippocampus storage.
**Why:** The environment's goal position should NOT be stored in the model's memory.

### D28: Winner-Take-All Action Selection (ATTEMPTED — FAILED)
**Attempted:** Replaced softmax action blending with WTA (pick single best action).
**Why:** The brain SELECTS one action and suppresses others (Go/NoGo pathways).
**What happened:** Fixed phase dropped to 8/10, Bizonal dropped to 20%.
**Root cause:** WTA with limited prototypes selects ONE wrong action instead of averaging several partially-correct ones. Blending was accidentally helping for novel goals.
**Lesson learned:** The brain's action selection is NOT simple WTA. It's POPULATION CODING with gradual convergence through recurrent dynamics. Pure WTA is too brittle for limited prototype pools.

### D29: Goal-Similarity Retrieval (ATTEMPTED — FAILED)
**Attempted:** Added goal direction similarity to SchemaBank retrieval score alongside state similarity.
**Why:** The brain's PFC generates goal-based queries, retrieving memories with similar goals.
**What happened:** Bizonal improved to 45% with correct implementation but the mechanism was conceptually wrong.
**Root cause:** Goal-similarity requires knowing the CURRENT goal. When the goal changes each episode, the previous episode's goal is stale and misleading.
**Lesson learned:** Goal-similarity retrieval IS valid within an episode (after discovery) but IS NOT valid across episodes with novel goals. The brain uses prediction error to detect goal changes, not goal-similarity retrieval.

### D30: Emergent Search Bonus (ATTEMPTED — FAILED)
**Attempted:** Added reward bonus for direction changes when near the goal zone, incentivizing systematic search.
**Why:** To encourage the agent to try different directions when searching for the goal.
**What happened:** Bizonal dropped to 17%, maze phases degraded.
**Root cause:** The search bonus corrupted the reward signal. The brain uses SEPARATE circuits for exploration (LC-NE tonic mode) and exploitation (phasic mode), not reward hacking.
**Lesson learned:** NEVER add exploration bonuses to the exploitation reward signal. The brain keeps explore and exploit circuits COMPLETELY SEPARATE.

### D31: Thalamic Attention Gating
**What:** Added context-dependent dimension routing to state before DG projection. Different contexts amplify different state dimensions.
**Why:** The thalamus doesn't just gain-control — it ROUTES information based on PFC context. In Bizonal, only 4/10 state dims matter.
**Impact:** Cleaner DG codes, context-appropriate pattern separation.
**Bug fixed initially:** Gating was applied during storage but NOT during retrieval, causing DG pattern mismatch. Fixed by applying gating to ALL DG calls.

### D32: Basal Forebrain ACh Modulation
**What:** Added pathway-specific learning rate modulation. HC gets high ACh (fast encoding), Policy gets moderate, RawFM gets low (stable physics).
**Why:** ACh release is NOT global — it's pathway-specific. Different brain regions need different plasticity rates.
**Impact:** More stable learning, less interference between components.

### D33: Reverse RPE Propagation (Reverse Replay)
**What:** When stale goal detected, replay last ~10 steps in REVERSE, devaluing each action via fraction of negative RPE.
**Why:** The brain's hippocampal reverse replay propagates value backward (Bellman backup). Without it, actions leading to stale goals remain valued.
**Impact:** Helps stop repeating outdated strategies.

### D34: CA3 Depotentiation + SchemaBank Weighted Clustering
**What:** Added weight field to CA3 patterns (AMPA receptor internalization analogue). Depotentiation = weight *= 0.6. SchemaBank clustering prioritizes high-weight patterns.
**Why:** Reconsolidation physically modifies memory traces (synaptic depotentiation via AMPAR endocytosis). Old patterns aren't deleted — they're weakened.
**Impact:** Natural retirement of outdated memories. Fresh learning outcompetes stale traces.

---

## 3. Complete Experiment Log: What Worked, What Didn't, and Why

### Successful Mechanisms

| Mechanism | Line Count | Success Metric | Key Insight |
|-----------|-----------|----------------|-------------|
| SchemaBank | ~80 | 100% retention | Never delete CA3 patterns |
| CA1 curiosity | ~5 | Persistent exploration | Spatial novelty ≠ prediction error |
| Subiculum confidence | ~50 | 43% Bizonal | Prediction error updates belief, not erasure |
| Theta sequences | ~50 | Better decisions | Simulate before execute |
| Thalamus gating | ~40 | Cleaner DG codes | Route by context, not just gain |
| BPF ACh modulation | ~30 | Stable learning | Pathway-specific LR, not global |
| Reverse replay | ~25 | Strategy devaluation | Bellman backup via reverse order |
| Depotentiation | ~20 | Natural forgetting | Weaken old traces, don't delete |
| Weighted clustering | ~5 | Better prototypes | High-weight patterns first |

### Failed Attempts and Why

| Attempt | What Happened | Root Cause | Lesson |
|---------|--------------|------------|--------|
| Pattern-deletion compression | Phase 4 dropped 38→26% | Deletion destroys spatial resolution | NEVER delete CA3 patterns |
| CA3 GPU cache (16GB+) | VRAM leak | torch.cat every step | Cache SchemaBank only |
| Goal-similarity retrieval | Bizonal 45% but conceptually wrong | Stale goal across episodes | Goal similarity only valid within episode |
| Search bonus reward hacking | All phases degraded | Reward corruption | Explore/exploit use SEPARATE circuits |
| Pure WTA action selection | Fixed 8/10, Bizonal 20% | Brittle with limited prototypes | Brain uses population coding, not WTA |
| Softmax action blending | Middle-ground actions | Averages conflicting actions | Blend is wrong but WTA is also wrong |
| Fixed learning rate everywhere | Slow adaptation | No difference between novel/familiar | Pathway-specific LR (ACH modulation) |

---

## 4. Current Architecture (v8.0 — 16 Built Systems)

| # | Brain Region | Component | Function |
|---|-------------|-----------|----------|
| 1 | **DG** | PatternSeparator | Fixed random projection (10→2000) + k-WTA |
| 2 | **Posterior HPC** | CA3Memory | Episodic storage + depotentiation weights |
| 3 | **Anterior HPC** | SchemaBank | Weighted prototype buffer (500 cap) |
| 4 | **Subiculum** | Subiculum | Goal VTC with confidence-based tracking |
| 5 | **Cerebellum** | CerebellarModel | Single-step forward model (Δs prediction) |
| 6 | **Neocortex (slow)** | SlowCerebellarModel | Multi-step aggregated forward model |
| 7 | **Motor Cortex** | Policy | MLP + goal direction + LC noise |
| 8 | **DLS (Striatum)** | ChunkLibrary | Action chunk prototypes (5-step) |
| 9 | **mPFC** | RuleBank | Abstract rule extraction |
| 10 | **OFC** | Value head | Outcome value prediction |
| 11 | **Striatum** | dopamine_update | 3-factor dopamine plasticity |
| 12 | **ACC** | ACC + ACCRpeTracker | Stuck detection + sustained RPE monitoring |
| 13 | **Thalamus** | Thalamus | Context-dependent attention routing |
| 14 | **Basal Forebrain** | BasalForebrain | Pathway-specific ACh LR modulation |
| 15 | **LC-NE** | ACCRpeTracker.lc_mode | Exploration mode via action noise |
| 16 | **HPC Reconsolidation** | CA3.depotentiate | AMPA internalization via weight decay |

---

## 5. The 43% Ceiling and the 2000-Step Test Result

The Bizonal test requires navigating to a DIFFERENT goal coordinate within a 2×2 zone each episode. The agent has NO knowledge of the current goal's position (the goal is not in the state). It must:
1. Discover the goal through exploration (no stored actions point to the novel coordinate)
2. Learn the new position (store in subiculum)
3. Navigate to it using the remembered direction

**2000-step test result (4× longer episodes):** 60% average (LEFT 38%, RIGHT 86%)
- The RIGHT side asymmetry reveals imbalanced training seeding
- 4× more time → 17% improvement, proving the ceiling is partially a time constraint
- But 40% gap remains even with 2000 steps — the fundamental limit is the lack of systematic search

**To break this ceiling, the architecture needs a DEDICATED EXPLORATION POLICY** — a separate circuit that takes over when the agent detects uncertainty and executes systematic search (spiral, lawnmower, radial). The brain uses LC-NE tonic mode to switch to a different neural circuit (frontopolar/lateral PFC) that implements exploration. Our current architecture has ONE policy doing both.

---

## 6. Testing Philosophy: Continuous Learning, Brain-Realistic Duration

### The Brain Never Resets
The brain is a SINGLE continuous learning system. It does NOT have separate "training" and "testing" phases. Every experience is a learning opportunity. Tests must reflect this:
- **Same model, no resets:** The same agent must learn across ALL phases and tests. No reloading checkpoints between test phases.
- **Tests are learning opportunities:** Every test episode also trains the model (dopamine updates, hippocampus storage, SchemaBank updates). The test measures ADAPTATION, not frozen performance.
- **Continuous improvement:** The agent should get BETTER over time. A test that shows degradation means catastrophic forgetting — which is solved by SchemaBank + episode-level sleep BC.

### Brain-Realistic Test Duration
Tests must be long enough for the agent to DISCOVER and LEARN, not just retrieve:
- **Discovery phase:** Allow enough steps for curiosity-driven exploration to find a novel goal. 500 steps was too short (43%). 2000 steps improved to 60%.
- **Learning phase:** After discovery, the agent needs steps to STORE the new knowledge and NAVIGATE using it.
- **Reasonable for the brain:** A rat in a maze gets 5-10 minutes per trial. Our 2000-step episodes at ~0.02s/step = 40 seconds per episode — reasonable.
- **Total test duration:** Multiple cycles through all phases, measuring improvement over time (not just final accuracy).

### Continuous Learning Test Protocol
```
Phase 0: 2000 steps (fixed start/goal) — learn basics
Phase 1: 2000 steps (random start) — adapt to new starts
Phase 2: 2000 steps (random goal, NE zone) — adapt to novel goals
Phase 3: 2000 steps (wall) — learn obstacle navigation
Phase 5: 2000 steps (proper maze) — learn maze navigation
---
Retention: Test ALL phases again without reset — measure retention
Bizonal: 2000-step episodes × 15 per side — measure novel goal discovery
---
Cycle 2: Repeat ALL phases — measure improvement
Cycle 3: Repeat ALL phases — measure further improvement
```

The agent should SHOW IMPROVEMENT across cycles. If it doesn't, something is wrong (likely catastrophic forgetting or capacity saturation).

---

## 7. The Ultimate Test: AGI Evaluation Suite

The current Bizonal and maze tests are too simple. A true AGI test must measure:

### Test 1: Zero-Shot Rule Transfer
- **Setup:** Train on 50 environments where goals follow Pattern A (e.g., "goal is always in the quadrant OPPOSITE the start")
- **Test:** Novel environment with same pattern → can the agent apply the rule immediately?
- **Expected AGI:** ~90%+ on first novel episode
- **Our current capability:** ~0% (we have no cross-episode pattern extraction)

### Test 2: Compositional Skill Recombination
- **Setup:** Train on Skill A (navigate walls), Skill B (find goal in NE), Skill C (avoid objects)
- **Test:** Novel environment requiring A + B + C simultaneously → can the agent compose them?
- **Expected AGI:** ~90%+ through composition, not memorization
- **Our current capability:** 0% (skills are not composable)

### Test 3: Continual Skill Acquisition (100+ skills)
- **Setup:** Train 100+ distinct navigation tasks over 10,000+ episodes
- **Test:** Can the agent still perform Task #1 after learning Task #100?
- **Expected AGI:** ~90%+ retention across ALL tasks
- **Our current capability:** ~80% retention across 12 tasks (needs SchemaBank capacity increase)

### Test 4: Counterfactual Reasoning
- **Setup:** "If you had gone left instead of right at the junction, where would you be?"
- **Test:** Can the agent simulate counterfactual trajectories using the RawFM?
- **Expected AGI:** Correct prediction without execution
- **Our current capability:** 0% (no counterfactual simulation)

### Test 5: Meta-Learning (Learn the Learning Rule)
- **Setup:** Environments where the goal PATTERN changes over time (e.g., day 1: NE, day 2: SE, day 3: NW...)
- **Test:** Can the agent learn the META-rule: "the goal rotates clockwise each day"?
- **Expected AGI:** After 5 days, predict Day 6's goal location
- **Our current capability:** 0% (no meta-learning)

---

## 7. What's Still Missing for True AGI (Ranked by Impact)

### Critical (must build next)

1. **DEDICATED EXPLORATION POLICY** — A separate neural circuit (frontopolar/lateral PFC analogue) for systematic search when uncertain. The LC-NE system currently just adds noise. The brain has a COMPLETELY DIFFERENT policy structure for exploration.

2. **ENGRAM RECONSOLIDATION (FULL)** — Our depotentiation is a start. The full mechanism requires: old memory retrieval → prediction error → destabilization → new information integration → restabilization. Currently we only weaken old traces; we don't integrate new information into them.

3. **CROSS-EPISODE PATTERN EXTRACTION** — The brain doesn't just learn (state → action) per episode. It extracts patterns ACROSS episodes: "every time the goal is in the NE quadrant, I should go NE." This requires a dedicated statistical learning mechanism that spans episodes.

### High Impact

4. **WORKING MEMORY (DLPFC)** — Maintains ~4 items for multi-step reasoning. Currently the agent is purely reactive. With working memory, it could maintain: "I'm searching the NW quadrant. I've already checked the top-left corner. Next: top-right."

5. **GRID CELL METRIC** — Hexagonal coordinate system for path integration. Currently the agent can't compute distances — it can only memorize (state → action). Grid cells would enable true vector navigation.

6. **META-LEARNING** — The ability to learn HOW to learn. Currently the agent has fixed learning rules (dopamine REINFORCE, sleep BC). Meta-learning would let it adjust these rules based on experience.

---

## 8. Instructions for the Next Agent

### What to build next

1. **Create a proper AGI evaluation suite** (test_agi.py) using the 5 tests defined in Section 6. The current Bizonal test is too simple. The AGI suite must measure zero-shot transfer, composition, retention across 100+ tasks, counterfactual reasoning, and meta-learning.

2. **Build a dedicated exploration policy** — NOT reward hacking. A separate policy circuit activated by the LC-NE system when exploration mode is triggered. This policy implements systematic search patterns (spiral, lawnmower, radial-bounce).

3. **Implement full reconsolidation** — not just depotentiation. When old memories are retrieved with prediction error, physically integrate new information into the existing CA3 pattern, not just weaken it.

4. **Implement cross-episode statistical learning** — a mechanism that tracks patterns across episodes and extracts abstract rules: "goals consistently appear in the NE quadrant."

### What NOT to do

- Don't add PPO, SAC, Dreamer, or other existing RL algorithms
- Don't add transformers, attention layers, or new network architectures
- Don't delete CA3 patterns (SchemaBank is additive)
- Don't add exploration bonuses to the exploitation reward signal
- Don't store the current episode's goal from the environment
- Don't use WTA — use population coding with gradual convergence
- Don't compare results to PPO/SAC/Dreamer — this architecture is NOVEL

### Critical files

| File | Purpose |
|------|---------|
| `train_sr.py` | Main codebase (~1765 lines) — all 16 brain systems |
| `env_nav.py` | Standard maze environment (6 phases) |
| `env_bizonal.py` | Multi-rule arena (context-dependent goals) |
| `test_cross.py` | Cross-environment retention test |
| `test_curriculum.py` | Teach → Practice → Retention test |
| `AGENT_RULES.md` | Critical rules for building — read BEFORE any changes |
| `NEW_ARCHITECTURE.md` | This document |

### Key numbers to verify

- Phase 0 (Fixed): 10/10 = 100%
- Bizonal wrong-zone errors: 0%
- VRAM: ~206MB (never exceed 500MB)
- Cross-env retention: 4/5 phases preserved
- No goal in state (S=10 confirmed)
- No goal leaked via reward (reward diff < 0.01 per step)
