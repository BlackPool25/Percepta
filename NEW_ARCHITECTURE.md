# Percepta — New Architecture Reference

**Version:** 1.0 (Redesign)  
**Date:** June 21, 2026  
**Purpose:** Complete reference for the redesigned brain-inspired architecture. Every agent should read this before making changes.

---

## 1. Project Vision

Build an AI agent that learns continuously from experience in a simulated world — without catastrophic forgetting, without pre-built backbones, and without being a benchmark-optimization project. The agent should:

- Start with innate priors (physics, object permanence, spatial geometry)
- Explore actively, driven by uncertainty-reduction curiosity
- Learn from few examples via one-shot Hebbian binding
- Consolidate important experiences slowly into stable long-term knowledge
- Revise beliefs when contradicted by new evidence
- Do all this without transformers, without frozen pretrained weights, and without unbounded memory growth

**This is NOT a benchmark-optimization project.** Benchmarks are sanity checks that individual mechanisms work, not the goal itself.

---

## 2. Core Principles

1. **Strict neuroscience fidelity.** Every mechanism must map accurately to a brain mechanism. If the mapping is aspirational rather than operational, the mechanism needs redesign or removal.

2. **Earn complexity one step at a time.** Build one mechanism, validate it in isolation, then add the next. Never build two untested mechanisms at the same time.

3. **No pre-built backbones.** No ResNet, no transformers, no foundation models. The architecture must work from first principles. "Frozen physics" priors (hardcoded rules) are acceptable.

4. **No classifier.** The brain does not have a dedicated "classification" module. Classification is an emergent behavior, not a system component.

5. **Prediction is the primary learning signal.** The neocortex learns by predicting sensory input at multiple levels of abstraction. Self-supervised prediction, not supervised classification.

6. **Epistemic honesty.** If a mechanism doesn't work, document why and remove it. Don't keep it because it was expensive to build.

7. **Efficiency matters.** No transformer-scale compute, no unbounded memory growth, no wasted computation.

---

## 3. Non-Negotiable Requirements

These are the 8 requirements derived from the brain's memory architecture. Every component must serve at least one requirement. No component that doesn't serve any requirement.

| # | Requirement | Brain Mechanism | Current Status |
|---|-------------|-----------------|----------------|
| 1 | Fast-write + slow-write memory split | Hippocampus (fast, one-shot) + Neocortex (slow, gradual) | DESIGNED |
| 2 | Sparse, decorrelated representations | Dentate Gyrus pattern separation | DESIGNED |
| 3 | Modular routing | Anatomical specialization (different regions for different content) | NOT STARTED |
| 4 | Replay mechanism (sleep consolidation) | Hippocampal sharp-wave ripples training cortex | NOT STARTED |
| 5 | Importance-weighted, protected updates | Synaptic tagging and capture hypothesis | NOT STARTED |
| 6 | Salience/novelty gating plasticity | Neuromodulation (dopamine, norepinephrine, acetylcholine) | NOT STARTED |
| 7 | Index-based, content-addressable retrieval | Hippocampal indexing (CA3 autoassociative memory) | DESIGNED |
| 8 | Genuinely online, non-IID training | Continuous streaming experience | NOT STARTED |

---

## 4. Architectural Decisions Log

All decisions recorded with rationale. If a decision is revisited, record the new decision and why the old one was wrong.

### D1: Fast system architecture (2026-06-21)
**Decision:** Modern Hopfield network with explicit pattern buffer storage, not classical Hopfield weight matrix.
**Why:** Higher capacity (exponential in pattern dimension), simpler to debug, naturally supports key-value separation, and the attention-based retrieval is differentiable for downstream gradient modulation.
**Neuroscience mapping:** CA3 autoassociative memory. The pattern buffer + attention approximates CA3's recurrent collateral dynamics for pattern completion.
**Source:** Step 1 validation.

### D2: Storage format (2026-06-21)
**Decision:** Pattern buffer (explicit storage), not weight matrix (outer product).
**Why:** Simpler implementation, higher capacity, avoids O(N²) storage cost, supports learnable temperature for retrieval modulation.
**Neuroscience mapping:** Moderate fidelity. Weight matrix (true Hebbian) would be more biologically accurate, but the pattern buffer approach retains the essential property: one-shot storage + content-addressable retrieval.

### D3: Learning rule for fast system (2026-06-21)
**Decision:** Hybrid — Hebbian storage (gradient-free, one-shot) + gradient-based retrieval modulation (learnable beta temperature).
**Why:** Storage must be fast and local (one presentation is enough). Retrieval benefits from learnable parameters (how sharp should pattern matching be?). This decouples the two concerns.
**Neuroscience mapping:** Hebbian plasticity for CA3 recurrent synapses (local, one-shot). Neuromodulatory tuning of retrieval dynamics (e.g., acetylcholine sharpening representations).

### D4: Pattern separation mechanism (2026-06-21)
**Decision:** Fixed random projection + k-WTA sparsity. The projection matrix P is never learned.
**Why:** Pattern separation is performed by the dentate gyrus via fixed, strong mossy fiber connections. There is no evidence the DG's pattern separation function is learned. Fixed random projection + k-WTA achieves the essential properties: sparsity, decorrelation, and similarity-preserving (similar inputs produce correlated but distinct sparse codes).
**Neuroscience mapping:** High fidelity. DG granule cells receive input from entorhinal cortex via perforant path (fixed), perform pattern separation via extreme sparsity (~0.5-5% active), and project to CA3 via mossy fibers (strong, sparse).

### D5: Fast/slow interaction (2026-06-21)
**Decision:** Full bi-directional loop. Slow system queries fast system during forward pass (pattern completion). Fast system replays to slow system during sleep (consolidation).
**Why:** The brain's hippocampal-cortical system is bi-directional. The cortex sends features to hippocampus for encoding. The hippocampus sends retrieved patterns back to cortex. During sleep, the hippocampus replays stored patterns to train the cortex.
**Status:** Forward direction (slow → fast query) is designed. Reverse direction (fast → slow replay) is not yet implemented.

### D6: Key-Value memory separation (2026-06-21)
**Decision:** Hopfield memory stores (key, value) pairs. Keys are DG-separated sparse codes. Values are original feature vectors.
**Why:** This separates addressing (retrieval cue) from content (retrieved information). The DG projection decorrelates the keys for reliable attention-based retrieval. The values preserve the original features for downstream use.
**Neuroscience mapping:** High fidelity. DG → CA3 provides pattern-separated keys (via mossy fibers). CA3 → CA1 provides feature retrieval (via Schaffer collaterals).
**Source:** Step 2 validation confirmed DG pattern separation is essential for handling correlated features.

### D7: No classifier (2026-06-21)
**Decision:** The architecture does not include a classification head, cross-entropy loss, or any label-based training.
**Why:** The brain does not have a dedicated classifier. Classification is an emergent behavior from prediction and action. Removing the classifier forces the architecture to learn useful representations via prediction alone, which is more biologically faithful.
**Implication:** The old Percepta classification benchmarks (Split-MNIST, CIFAR) are not applicable to this architecture. New evaluation metrics are needed based on prediction error, retrieval accuracy, and goal-directed behavior.

### D8: Controlled-correlation synthetic data for validation (2026-06-21)
**Decision:** Use synthetic data with controlled correlation structure (clustered features) for Step 2 validation, not MNIST or other real datasets.
**Why:** Allows precise measurement of how correlation structure affects memory capacity. Avoids confounding factors from real data (unknown correlations, label structure).
**Source:** Step 2 validation.

---

## 5. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SLOW SYSTEM (Neocortex)                      │
│  Gradient-based, gradual learning, predictive/sensory processing │
│                                                                  │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐   │
│  │ Sensory │──▶│  Feature  │──▶│  Action  │──▶│ Environment  │   │
│  │ Input   │   │  Extractor│   │  Output  │   │  (world)     │   │
│  └─────────┘   └────┬─────┘   └──────────┘   └──────┬───────┘   │
│                     │                                │           │
│                     │ features (forward pass)         │ sensory   │
│                     ▼                                │ feedback  │
│              ┌──────────────┐                        │           │
│              │   CA1/EC     │◀────── replay ──────── │           │
│              │ (integration) │     (sleep)            │           │
│              └──────┬───────┘                        │           │
│                     │ retrieved features              │           │
└─────────────────────┼─────────────────────────────────┘           │
                      │                                            │
┌─────────────────────┼─────────────────────────────────┐           │
│                     ▼     FAST SYSTEM (Hippocampus)    │           │
│  Hebbian, one-shot, separate from gradient flow       │           │
│                                                        │           │
│  features ──▶ DG: PatternSep(h) ──▶ sparse key z       │           │
│                  (fixed projection + k-WTA)            │           │
│                                                        │           │
│  z ──▶ CA3: Store (z, h) in pattern buffer            │           │
│        (one-shot Hebbian, no gradient)                 │           │
│                                                        │           │
│  z_q ──▶ CA3: Retrieve h* via attention over keys     │           │
│           h* = softmax(β · z_q · Z_keys^T) · H_values │           │
│                                                        │           │
│  h* ──▶ CA1: Compare h* vs h → novelty signal         │           │
│           novelty = ||h* - h|| → neuromodulatory gate │           │
│                                                        │           │
│  During SLEEP:                                         │           │
│  CA3 replays stored patterns sequentially              │           │
│  → CA1 → EC → cortex learns from replay               │           │
│                                                        │           │
└────────────────────────────────────────────────────────┘           │
                                                                     │
                      SLEEP CONSOLIDATION                            │
  ┌─────────────────────────────────────────────────────────┐        │
  │  CA3 → CA1 → EC: Sequential replay of stored patterns   │        │
  │  Cortex trains on replayed patterns (gradient updates)   │        │
  │  Active forgetting: low-importance patterns decay        │        │
  │  Importance-weighted protection: frequently replayed     │        │
  │  patterns are protected from overwrite                   │        │
  └─────────────────────────────────────────────────────────┘        │
```

### Data Flow Summary

**Forward pass (perception):**
1. Sensory input → Feature Extractor → features h
2. h → DG: z = PatternSeparator(h)
3. z → CA3: store (z, h) (one-shot)
4. z_query → CA3: retrieve h* = softmax(β · z · Z^T) · V
5. h* → CA1: novelty = ||h* - h||
6. h* → integration → action

**Sleep (consolidation):**
1. Cortex → Hippocampus: trigger replay
2. CA3 retrieves stored patterns sequentially
3. Each pattern sent to Cortex via CA1 → EC
4. Cortex trains on replayed patterns (prediction + consolidation)
5. Low-importance patterns decay

---

## 6. Component Specifications

### 6.1 Pattern Separator (DG)

**File:** `hopfield_memory.py` → `PatternSeparator`

**Input:** features h (..., d_in)
**Output:** sparse binary code z (..., d_hidden), sparsity fraction active

```
z = k-WTA( h @ P )  where P is fixed random projection
```

**Parameters:**
- `input_dim`: feature dimension (e.g., 128)
- `hidden_dim`: sparse code dimension (e.g., 2000)
- `sparsity`: fraction of units active (tested: 0.005 to 0.10, optimal: 0.01-0.02)
- `P`: fixed random projection matrix (He initialization, never learned, gradient disabled)

**Properties validated:**
- [x] Random projection + k-WTA produces decorrelated sparse codes from random inputs (Step 1)
- [x] DG pattern separation is essential for handling correlated features (Step 2)
- [ ] Works with learned features from trained encoder

### 6.2 Hopfield Memory (CA3)

**File:** `step2_memory_with_correlation.py` → `KeyValueHopfieldMemory`

**Input:** key (sparse code z), value (feature h)
**Output:** retrieved value h* via attention over stored keys

```
Storage: append (z, h) to (keys list, values list) — one-shot, no gradient
Retrieval: h* = softmax(β · z_q · Z^T) · V
```

**Parameters:**
- `key_dim`: sparse code dimension (e.g., 2000)
- `value_dim`: feature dimension (e.g., 128)
- `beta`: temperature (learnable, initialized to 1.0)

**Properties validated:**
- [x] Perfect retrieval (F1=1.0) for random decorrelated patterns up to capacity (Step 1)
- [x] Near-perfect retrieval (cosim > 0.998) for structured correlated patterns up to 500 patterns (Step 2)
- [x] DG pattern separation is critical: flat keys (no DG) degrade to cosim 0.52 at high spread
- [ ] Memory stability under key drift (stored keys change over time)
- [ ] Capacity limit under adversarial conditions (highly overlapping patterns)

### 6.3 Key-Value Separation (CA3→CA1)

**Status:** Designed, validated in Step 2.

The separation of addressing (keys = sparse codes) from content (values = feature vectors) is critical. The keys are pattern-separated by DG for reliable retrieval. The values preserve the original features for downstream use.

### 6.4 Novelty Detection (CA1)

**Status:** Not implemented.

Compare retrieved value h* with current input h. Novelty = ||h* - h|| or cosine distance. This signal gates plasticity in the slow system (neuromodulation).

**Implementation plan:**
- After retrieval, compute novelty = 1 - cosine_similarity(h*, h)
- If novelty > threshold: "this is new/surprising" → modulate learning rate
- If novelty < threshold: "this matches stored memory" → suppress learning
- This is the neuromodulatory gate (requirement 6)

**Not yet built.**

### 6.5 Sparse Autoencoder (Cortex, preliminary)

**Status:** Not implemented. Planned for Step 3b/4.

The slow system's initial form is a sparse autoencoder:
- Encoder: input → features h (with sparsity regularization)
- Decoder: features h → reconstructed input
- The features h feed into the Hopfield memory via DG
- Memory retrieval produces h* which is decoded instead of h

**Learning:** Reconstruction loss (MSE) + sparsity penalty on features.

### 6.6 Predictive Cortex (Neocortex, full)

**Status:** Not implemented. Planned for Step 5+.

The full slow system learns by predicting sensory input across multiple timescales and hierarchies. This is deferred until the memory and basic autoencoder are validated.

### 6.7 Sleep Consolidation

**Status:** Not implemented.

Mechanism for transferring fast memory patterns into slow system:
1. During sleep/rest, CA3 replays stored patterns sequentially
2. Each pattern (key, value) → slow system trains on the value
3. Slow system learns to produce the same features as the fast system retrieved
4. Over many sleep cycles, knowledge transfers from fast to slow

**Design considerations:**
- Frequency-based replay: more important patterns are replayed more often
- Spaced repetition: patterns replayed at increasing intervals (like human memory)
- Interleaving: random order prevents sequential bias
- Active forgetting: patterns that are never replayed decay

### 6.8 Modular Routing (MoE)

**Status:** Not implemented. Planned for Step 6+.

Mixture-of-Experts gating to route different content types to different sub-networks. Deferred until the core memory and perception systems are validated.

### 6.9 Genuinely Online Training

**Status:** Not implemented. The current training uses batch-based optimization. The online protocol will process one sample at a time without shuffling.

---

## 7. Step-by-Step Validation Plan

### Step 1: Standalone Hopfield Memory ✓

**Goal:** Validate that modern Hopfield (attention-based) retrieval works for decorrelated patterns.

**Test:** Generate random sparse patterns, store them, corrupt and retrieve. Measure F1 score.

**Results:**
- 2% sparsity (40/2000 active): F1=1.000 for k up to 500 at 50% corruption
- 0.5% sparsity (10/2000 active): Fails at k=150 (too few active bits → high overlap)
- 1-2% sparsity: sweet spot for N=2000
- Robustness: F1=1.000 up to 86% corruption, fails at 95% (2 active bits remaining)

**Conclusion ✅:** Pattern separation + Hopfield retrieval works perfectly for decorrelated patterns.

### Step 2: Memory with Structured Correlations ✓

**Goal:** Validate that memory handles correlated features (as real learned embeddings will).

**Test:** Generate clustered features with controlled within-cluster variance. Store with DG-separated keys vs. flat (no separation) keys.

**Results:**
- DG-separated keys: cosim > 0.998 across ALL correlation levels (std 0.01 to 2.0)
- Flat keys (no DG): cosim degrades from 0.80 to 0.52 as correlation decreases
- DG keys handle 500 patterns in 50 clusters with cosim > 0.998
- DG pattern separation is ESSENTIAL for reliable retrieval with correlated features

**Conclusion ✅:** Key-value Hopfield memory with DG pattern separation handles structured features robustly.

### Step 3a: Memory Stability Under Key Drift ✓

**Goal:** Validate memory remains stable when stored patterns' keys drift over time (as happens when a learned encoder's features shift during training).

**Test:** Store patterns from clustered features. Gradually drift ALL features via random walk + decay. At each step, query using drifted key, measure retrieval cosim against original stored values.

**Results:**
- After 200 steps with cumulative drift of 0.16: cosim=0.997, key overlap=83%
- Drift rate has negligible effect (0.0 to 0.1 all converge to cosim > 0.997)
- Retrieval remains near-perfect because DG's k-WTA preserves top active units under moderate feature perturbation

**Critical finding:** Gradual drift is handled well. The old RSSM failure was caused by REPRESENTATION COLLAPSE (all features converging to the same point), NOT by drift. A sparse autoencoder's reconstruction loss prevents collapse.

**Conclusion ✅:** The DG + Hopfield memory is inherently robust to gradual feature drift. The failure mode is collapse, not drift.

### Step 3b: Sparse Autoencoder Integration ✓

**Goal:** Validate that the Hopfield memory works end-to-end with a learned encoder.

**Test:** Train CNN sparse autoencoder on 64×64 geometric shape images (circle, square, triangle). Features h (128-dim) → DG (2000-dim, 2% sparse) → Hopfield memory → retrieve h* → decode → reconstruction. Use key-space corruption (50% active units zeroed) for pattern completion test.

**Results:**
- Autoencoder: recon MSE 0.251 → 0.002 (100× reduction). Feature std=0.24-0.27. NO COLLAPSE.
- Direct memory (no corruption): MSE=0.00330 (identical to autoencoder baseline)
- Retrieved (50% key corrupt): MSE=0.00355, feature cosim=0.9999
- Capacity with learned features: cosim=1.000 up to 200 patterns

**Critical finding:** The sparse autoencoder does NOT collapse (unlike the old RSSM), and the memory works perfectly with learned features. The end-to-end pipeline is validated.

**Conclusion ✅:** The full pipeline (encoder → DG → Hopfield memory → decoder) works with learned features and near-perfect pattern completion.

### Step 4: Sleep Consolidation ✓

**Goal:** Validate that offline replay transfers knowledge from fast (memory) to slow (autoencoder) system.

**Design:** Store (image, key, value) triples during waking. During sleep, sample triples and train autoencoder with two losses:
1. Standard autoencoder loss on replayed images (encoder + decoder)
2. Decoder-only consolidation: decoder learns from memory-retrieved features (decoder only, no encoder gradient)

**Critical design lesson:** Initial approach used embedding matching loss (forcing encoder to match stale stored features), which caused feature collapse (std 0.26 → 0.09). The fix: only train the decoder from memory features; the encoder learns from replayed images via standard reconstruction. This avoids encoder-memory conflict.

**Results:**
- Recon MSE: 0.003 → 0.00127 (2.4× improvement)
- Feature std: 0.26 → 0.31 (increased diversity — opposite of collapse)
- After memory ablation: MSE=0.00140 (knowledge retained in autoencoder weights)
- Both loss components (recon + memory consolidation) decreased during sleep

**Conclusion ✅:** Sleep consolidation via image replay + decoder-only memory consolidation works. The autoencoder retains improved knowledge even when the memory is cleared.

### Step 5: Bi-directional Loop / CA1 Novelty Detection ✓

**Goal:** Validate that the CA1 novelty detection mechanism can distinguish familiar from novel patterns.

**Design:** During forward pass, compute both the actual features h (from encoder) and the memory-retrieved features h* (from memory using the DG-projected key). CA1 computes: novelty = 1 - cosim(h, h*). Low novelty = familiar (memory correctly recalled), high novelty = novel (input differs from any stored pattern).

**Results:**
| Category | Mean novelty | Z-score (vs familiar) | Interpretation |
|----------|-------------|----------------------|----------------|
| Familiar | 0.0000 | 0.0 | Perfect retrieval |
| Rotated (circle) | 0.055 | 552 | Small — rotation interpolation artifacts |
| Rotated (square) | 0.067 | 667 | Small — similar to circle |
| Rotated (triangle) | 0.048 | 479 | Small — similar to other shapes |
| New shape (diamond) | 0.293 | 2932 | Moderate — partial feature overlap with squares |
| Unseen position | 0.359 | 3589 | Highest — fully novel position context |

**Key findings:**
1. **Rotation is NOT learned invariance.** All three shapes have similar novelty (~0.05). The difference between rotation novelty and familiar novelty is caused by interpolation artifacts in the rotation transform, not by the encoder learning rotation-invariant features. The user's hypothesis about shape symmetry (circles = 0, squares = moderate, triangles = highest) was tested and disproven.
2. **Novelty is genuinely graded.** The z-score spans three orders of magnitude: rotation (z≈500) << new shape (z≈3000) ≤ position (z≈3600). This is a continuum, supporting proportional (graded) neuromodulation rather than binary novelty/familiar.
3. **Adaptive thresholds are required.** The familiar distribution has zero variance (exact retrieval due to decorrelated DG keys). Absolute thresholds break under this condition. The z-score approach with a minimum std floor (e.g., `fam_std = max(observed_std, 1e-4)`) generalizes across encoder settings and datasets.

**Conclusion ✅:** CA1 novelty detection works as a reliable graded novelty discriminator. Thresholds must be defined relative to the familiar distribution (z-score), not as absolute values.

### Step 6: Modular Routing (DEFERRED — not yet needed)

**Status:** Deferred. The single autoencoder handles all shape types (circle, square, triangle, diamond) without measurable interference. There is no demonstrated cross-type interference problem at current scale. Will be revisited if capacity limits are encountered with more diverse data.

### Step 7: Online Streaming ✓

**Goal:** Validate that the entire system works in a single-pass, non-IID streaming setting with distribution shifts and no task boundaries.

**Design:** Stream of 5000 individual shape images with 6 phased distribution shifts (center positions → extreme positions → center + diamonds → mixed all → back to center → extreme again). Each experience processed one at a time. Autoencoder updates on sliding windows (last 32 experiences, no shuffle). Memory with bounded capacity (200 patterns, FIFO eviction). Novelty computed as reconstruction error (predictive coding).

**Critical design evolution:**
- Initial approach used attention-based novelty (max attention weight + key logits). Collapsed after ~200 patterns due to blending (sufficient coverage means every query finds SOME weighted combination of stored patterns).
- Switched to reconstruction error novelty. This directly measures what the model doesn't understand, independent of memory contents. Never collapses because there is always some prediction error.
- Memory with bounded capacity (200) + FIFO eviction prevents indefinite growth. At capacity, the memory acts as a sliding window of diverse experiences.

**Results:**
- Novelties remain alive for all 5000 steps (range 0.04-0.25)
- Every distribution shift creates a measurable novelty change
- Model adapts online: reconstruction error decreases within each phase, rises at each shift
- Knowledge persists across shifts: "extreme again" at step 4500 shows novelty 0.04-0.07 vs first extreme phase at 0.25 (model retained knowledge across 3500 intervening steps)
- Memory saturates at 200 capacity and stays there
- Evidence of online continual learning without catastrophic forgetting

**Key insight:** Reconstruction error novelty is the correct approach for streaming. It is memory-independent, responsive to distribution shifts, and never saturates. Attention-based novelty collapses at scale.

**Conclusion ✅:** The system operates continuously in a single-pass streaming setting with distribution shifts. Novelty detection via reconstruction error provides robust online adaptation.

### Step 8: Curiosity-Driven Exploration ✓

**Goal:** Validate that novelty-based intrinsic motivation drives effective exploration.

**Design:** Agent with autoencoder + Hopfield memory + CA1 novelty detector processes a mixed stream (20% familiar, 40% novel position, 40% new shape). Curiosity decision: store patterns in memory proportionally to their novelty (attention-based + key similarity). Adaptive threshold (percentile-based) determines which patterns to store.

**Results:**
| Pattern type | Storage rate | Behavior |
|---|---|---|
| Familiar | 10% | Correctly ignored |
| New shape | 28% | Selectively stored |
| Novel position | 53% | Most stored (highest novelty) |

The agent preferentially stores novel patterns over familiar ones by 5:1. Graded response: position novelty > shape novelty > familiarity. Novelty signal maintains meaningful variance (std=0.302) as memory grows.

**Conclusion ✅:** Curiosity-driven exploration via novelty-based storage prioritization works. The agent selectively stores novel experiences and ignores familiar ones.

### Step 9: Embodied Learning (MUJOCO RUNNING)

**Goal:** Validate the complete agent in a 3D simulated environment (MuJoCo PointMaze).

**Status:** Three-phase training loop integrated and running. Sleep consolidation confirms. 100K-step comparison experiment ready.

**Results (preliminary, 5K steps):**
- RL loop verified: PPO updates, no crashes
- Sleep consolidation: encoder updates on memory replay, reduces reconstruction error in visited regions
- Fast novelty: temporal event-boundary detection — varies 0.0003-0.1284 as agent navigates
- Slow novelty: decreases from 0.0615 to 0.0021 as encoder consolidates visited regions
- Store rate: 15% steady-state (adaptive OR-gate thresholds)
- Memory size: 200 (FIFO capacity)

**Open question:** Does the exploration gradient (higher novelty in unvisited regions) actually pull the agent toward novel areas faster than random exploration? Requires 100K-step comparison against plain PPO baseline with trajectory coverage heatmaps.

**Comparison script:** `run_comparison.py` — runs Percepta (CLS) vs Plain PPO on PointMaze UMaze, logs trajectories and heatmaps at 2K-step intervals.

---

## 8. Results Summary

### Successfully Validated

| Component | Test | Metric | Result | Date |
|-----------|------|--------|--------|------|
| DG Pattern Separation | Random pattern completion | F1 | 1.000 for k up to 500 | 2026-06-21 |
| DG Pattern Separation | Robustness to corruption | F1 | 1.000 for corruption up to 86% | 2026-06-21 |
| Pattern Separator | Sparsity sweep | F1 | Optimal: 1-2% sparsity | 2026-06-21 |
| Key-Value Hopfield Memory | Clustered feature retrieval | Cosim | > 0.998 for all correlations | 2026-06-21 |
| Key-Value Hopfield Memory | Cluster count sweep | Cosim | > 0.998 for 1-50 clusters | 2026-06-21 |
| DG Pattern Separation | Value vs. flat keys | Cosim | DG: 0.999+, Flat: 0.52-0.80 | 2026-06-21 |
| Hopfield Memory | Key drift stability | Cosim | 0.997 after drift of 0.16 | 2026-06-21 |
| Sparse Autoencoder | Learn features without collapse | h_std | 0.24-0.27 (healthy) | 2026-06-21 |
| Sparse Autoencoder + Memory | End-to-end retrieval | Cosim | 0.9999 at 200 patterns | 2026-06-21 |
| Sparse Autoencoder + Memory | Pattern completion (50% key corrupt) | MSE | 0.00355 vs 0.00330 baseline | 2026-06-21 |
| Sleep Consolidation | Image replay + decoder memory training | Recon MSE | 0.003 → 0.00127 (2.4×) | 2026-06-21 |
| Sleep Consolidation | Feature collapse check | h_std | 0.26 → 0.31 (no collapse) | 2026-06-21 |
| Sleep Consolidation | Memory ablation | Recon MSE | 0.00140 (knowledge retained) | 2026-06-21 |
| CA1 Novelty Detection | Familiar vs. novel patterns (attention-based) | Novelty ratio | Rotation: 500×, Shape: 3000×, Position: 3600× | 2026-06-21 |
| CA1 Novelty Detection | Reconstruction error as novelty | MSE range | 0.04-0.25 over 5000 steps (never collapses) | 2026-06-21 |
| Curiosity Exploration | Storage rate by novelty type | Storage % | Familiar 10%, New shape 28%, Novel pos 53% | 2026-06-21 |
| Online Streaming | Single-pass non-IID, distribution shifts | Novelty by phase | All 6 phase transitions detected | 2026-06-21 |
| Online Streaming | Knowledge retention across shifts | Novelty at revisit | 0.04 vs 0.25 (retained knowledge) | 2026-06-21 |

### Not Yet Tested / Deferred

| Component | Test | Status |
|-----------|------|--------|
| Hopfield Memory | Capacity limit | Not found at current scale |
| Modular Routing (MoE) | Interference prevention | DEFERRED — no demonstrated interference between shape types |
| Bi-directional Loop | Full query-retrieval-consolidation | Integrated across Steps 4-5-7-8 |
| Embodied Learning | MuJoCo 3D agent | NEXT — see Section 12 |

---

## 9. Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `hopfield_memory.py` | Step 1: Standalone Hopfield memory test | `PatternSeparator`, `ModernHopfieldMemory`, test functions |
| `step2_memory_with_correlation.py` | Step 2: Memory with correlated features | `KeyValueHopfieldMemory`, `generate_clustered_features`, test functions |
| `step3a_key_drift_test.py` | Step 3a: Memory stability under key drift | Drift simulation + retrieval quality evaluation |
| `step3b_sparse_autoencoder.py` | Step 3b: Sparse autoencoder + memory integration | `SparseAutoencoder`, shape image generator, memory retrieval test |
| `step4_sleep_consolidation.py` | Step 4: Sleep consolidation | `TripleMemory`, `sleep_consolidation`, consolidation evaluation |
| `step5_novelty_detection.py` | Step 5: CA1 novelty detection | `NoveltyDetector`, familiarity vs. novelty discrimination test |
| `step7_online_streaming.py` | Step 7: Online streaming | `OnlineStream`, `OnlineAgent`, `BoundedMemory`, phased distribution shifts |
| `step8_curiosity_exploration.py` | Step 8: Curiosity-driven exploration | `CuriosityDrivenAgent`, `ExperienceSpace`, novelty-based storage |
| `NEW_ARCHITECTURE.md` | This document — complete architecture reference | — |

---

## 10. Open Questions

1. **Key drift tolerance.** When patterns are stored and the encoder's features drift over time, the stored keys become stale. How much drift can the memory tolerate before retrieval fails? This was the root cause of the old RSSM failure and must be answered before adding a learned encoder.

2. **Beta learning dynamics.** The temperature parameter beta is learnable via gradient from downstream tasks. What are the dynamics of this learning? Does beta converge, oscillate, or grow without bound?

3. **Storage capacity limit.** With 2% sparsity at N=2000, we haven't found the capacity limit for decorrelated patterns. Where is the actual boundary? More importantly, what is the effective capacity for structured patterns with real-world correlations?

4. **Forgetting mechanism.** Currently, the memory stores every pattern indefinitely. The brain actively forgets. What determines which patterns to retain and which to discard?

5. **Encoder-memory co-training.** When the encoder trains, the stored patterns' values become stale (the encoder's features change). But the keys also become stale. How do we handle simultaneous drift in both keys and values?

6. **Sleep protocol.** When does sleep occur? What triggers it? How many replay iterations per stored pattern? What determines replay order?

7. **Hard attention.** The current softmax attention blends ALL stored patterns. The brain retrieves one specific memory, not a blend. Should we use hard attention (retrieve the single best match)? This would more closely match pattern completion behavior.

---

## 11. Relationship to Old Architecture

The old Percepta architecture (v1.1 through v6.0) was a continual learning classifier with brain-inspired features bolted on. The new architecture discards:

| Old Component | Fate | Reason |
|---------------|------|--------|
| SlowCNN classifier | Removed | No classifier in brain. Classification is emergent. |
| FastDecayPath | Replaced | Gradient-based fast weights are not hippocampal. Replaced with Hebbian Hopfield memory. |
| Two-stage promotion gate | Removed | Gate was redundant compute — value was never proven over plain replay. |
| Core-set buffer | Replaced | Replaced with key-value Hopfield memory (content-addressable retrieval, not class-based FIFO). |
| Fisher importance mask | Removed | Proven ineffective (0% retention at 25% per-layer freeze). |
| Shadow model copy | Removed | Heavy-weight validation is not how consolidation works. |
| RSSM / JEPA | Deferred | Representation collapse killed it. Will be re-approached after memory and autoencoder are stable. |
| Actor-critic RL | Deferred | Cannot work until the perception-memory system is stable. |
| MuJoCo environment | Kept but deferred | Target environment for final embodied learning. |

---

## 12. Honest Assessment: What Percepta Is and Isn't

### What Percepta Genuinely Gets Right

1. **Complementary Learning Systems architecture.** Fast one-shot binding into a separate store, slow statistical learning in distributed weights, replay during "sleep" consolidating fast into slow. The held-out streaming test proves specific retention without catastrophic forgetting — the McClelland/O'Reilly CLS framework working as the neuroscience literature describes it.

2. **Sparse coding + pattern separation.** The DG-like k-WTA projection decorrelates inputs for reliable content-addressable retrieval. Validated across random patterns, structured correlations, and learned embeddings.

3. **Dual novelty signals.** Reconstruction error (slow system, predictive coding) + event-boundary detection (fast system, CA1-like comparison). Validated with separate thresholds and OR-gate storage logic.

### What's Abstracted Away (Principled Engineering Tradeoffs)

1. **Modern Hopfield attention for CA3 dynamics.** Pattern completion from partial cues is preserved; the mechanism is attention instead of recurrent collateral dynamics. The computational claim is the same.

2. **Reconstruction loss for prediction error.** MSE between input and reconstruction stands in for hierarchical predictive coding. It approximates the outcome without the mechanism.

3. **Gradient descent for slow system learning.** The brain doesn't use backprop. This is an open problem in neuroscience (credit assignment). Percepta doesn't solve it.

### What's Genuinely Missing

| Missing Component | Brain Function | Why It Matters for MuJoCo |
|---|---|---|
| Hierarchical predictive coding | Neocortex: each layer predicts the layer below | Flat autoencoder can't separate "I'm in a kitchen" from "I see a cup" |
| Working memory / PFC | Maintaining task-relevant info, goal-directed behavior | Policy head is a thin MLP where the brain has a specialized executive system |
| Basal ganglia action gating | Selecting which action to execute | No gating mechanism for action selection |
| Neuromodulation (DA, NE, ACh) | Timescale-specific plasticity modulation | Dual novelty is a coarse approximation of dopamine's role |
| Temporal structure / oscillations | Theta/gamma binding, STDP | System processes discrete frames with no temporal binding |
| Cerebellar forward model | Predicting sensory consequences of motor commands | No motor prediction for precise control |
| Credit assignment without backprop | How synapses learn without global error signals | Open neuroscience problem; not addressed here |

### Summary

Percepta is a solid engineering implementation of the Complementary Learning Systems theory — one of the most well-validated computational theories of hippocampal-cortical interaction. The sparse coding, pattern separation, sleep consolidation, and dual novelty signals are real neuroscience principles, not just analogies.

CLS is one theory about one subsystem. The full brain also does hierarchical predictive processing, neuromodulation across multiple timescales, executive control with working memory, motor prediction, temporal binding via oscillations, and credit assignment without backprop — none of which Percepta models. You've built the hippocampal-neocortical memory system. You haven't built a brain.

---

## 13. MuJoCo Integration Plan

### Design Decisions

1. **Policy architecture:** Small action head (Linear 128 → action_dim + tanh) + value head (Linear 128 → 1) on frozen encoder output. Policy gradients (PPO) update only the policy/value heads — the encoder is frozen during initial RL training to avoid destabilizing the representations that fast memory and novelty depend on.

2. **Intrinsic reward:** `R_int = α · fast_novelty + (1-α) · slow_novelty`, where fast novelty is hamming distance to nearest stored key and slow novelty is reconstruction error. Both components logged separately throughout training.

3. **Extrinsic reward:** Task-specific goal reward added to intrinsic reward for total RL signal.

4. **Memory storage:** Separate thresholds with OR logic (either signal above its own percentile threshold triggers storage). Same design validated in Step 7 dual novelty.

5. **Sleep consolidation:** Applied periodically during training, same mechanism as Step 4.

### Implementation Steps

1. **Observation encoder:** The existing sparse autoencoder takes 64×64 RGB frames from MuJoCo. Encoder output (128-dim) serves as state representation for the policy head.

2. **Policy + value heads:** Two Linear(128, action_dim) and Linear(128, 1) heads. Action head has tanh output for bounded continuous control.

3. **Training loop:** Per step:
   a. Encode observation → h
   b. Forward through memory + novelty → fast/slow novelty
   c. Compute action from policy head
   d. Step environment → next observation + extrinsic reward
   e. Total reward = extrinsic + alpha * fast_novelty + (1-alpha) * slow_novelty
   f. Store (obs, action, reward) in PPO buffer
   g. Periodically update policy via PPO

4. **Memory and consolidation:** Memory stores (key, value) pairs from experienced observations. Sleep consolidation runs periodically (every N episodes), replaying stored patterns through the autoencoder.

### Risks

1. **Novelty oscillation:** The agent may learn to shuttle between states to maximize intrinsic reward. Mitigation: log fast/slow components separately; clip intrinsic reward magnitude; lengthen FIFO eviction window if looping appears.

2. **Intrinsic reward scale:** Fast and slow novelty have different scales in 3D. α may need tuning, or use adaptive normalization (z-score each signal against its recent history).

3. **Encoder freeze vs. fine-tune:** Freezing prevents destabilization but limits adaptation. Plan to unfreeze gradually after the agent achieves basic competence.
