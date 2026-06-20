# Percepta — Complete Project Reference

**Version:** v6.0 (ResNet backbone)  
**Date:** June 20, 2026  
**Purpose:** Full reference for project continuation. Every agent should read this before making changes.

---

## 1. Project Vision

### The Original Idea

Build an AI agent that learns continuously from experience in a simulated world — without frozen weights, without transformer-scale cost, and without catastrophic forgetting. The agent should:

- Start with basic knowledge about how the world works (physics priors, object permanence, gravity)
- Explore its environment actively, driven by curiosity (uncertainty reduction, not raw novelty)
- Learn new concepts from few examples, like a human
- Store experiences temporarily (hippocampus analogue) and consolidate important ones into long-term memory (cortex analogue)
- Update or "forget" beliefs when contradicted by new evidence (recall-destabilization)
- Do all this efficiently (no transformer-scale compute, no unbounded memory growth)

### Core Philosophy

This is NOT a benchmark-optimization project. The goal is NOT to achieve 99% on Split-MNIST. The goal is to build a brain-inspired learning architecture. Benchmarks are used to test whether individual mechanisms work, not to maximize a number.

The project sits on open research frontiers. The combination of:
- Fast/slow weight split with staged consolidation gate
- Recall-destabilization modeled on systems reconsolidation
- Episodic retrieval for one-shot learning
- Uncertainty-reduction curiosity reward

Has no published equivalent in any single system. The project is exploring genuinely unknown territory.

### What "Beating SOTA" Actually Means

On standard continual learning benchmarks (Split-MNIST, CIFAR, Permuted MNIST), SOTA methods like DER/DER++ achieve 85-92%. Our current architecture achieves 79% on Split-MNIST. We are NOT ahead on these benchmarks.

The claim to beating SOTA is on DIMENSIONS THAT NO BENCHMARK CURRENTLY MEASURES:

| Dimension | SOTA | Percepta's bet |
|-----------|------|----------------|
| Noisy/redundant data filtering | None — all data treated equally | Stage 1 gate filters noise by frequency and persistence |
| Online adaptation speed | Standard replay buffers need batches | Fast-weight layer adapts within a single pass |
| Revising committed beliefs | No method does this | Recall-destabilization with 3-criteria detection |
| Episodic one-shot recall | Not present | Episodic retrieval at inference from stored memories |
| Priority-based computation | Uniform replay | Priority replay focuses compute on degrading clusters |

If Percepta succeeds, it won't beat SOTA on existing benchmarks — it will solve problems that existing SOTA methods don't even attempt.

---

## 2. Neuroscience Grounding

Every component in the architecture maps to a brain mechanism. Understanding the mapping is essential for making design decisions.

### Complementary Learning Systems (CLS) Theory

The brain has two complementary memory systems:

| Brain Component | AI Analogue | Status |
|----------------|-------------|--------|
| Hippocampus (fast encoding, sparse, episodic) | Episodic buffer + Fast-weight adapter | Built |
| Cortex (slow consolidation, dense, semantic) | Slow CNN/ResNet with replay protection | Built |
| Hippocampal indexing (pointer to cortical patterns) | Episodic retrieval + core-set probe sets | Built |
| Multi-stage molecular cascade (promote/demote) | Two-stage promotion gate | Built |
| Synaptic consolidation (importance-weighted protection) | EWC-style Fisher mask (vestigial) | Built but ineffective |
| Systems reconsolidation (recall destabilizes, requires restabilization) | Recall-destabilization with partial degradation | Prototyped |
| Metaplasticity (plasticity scales with contribution history) | Fast-decay-path stability tracking | Built |
| Pattern separation (DG) / Pattern completion (CA3) | Three fast paths (general/specific/residual) | Built |
| Prediction error as learning signal | KL divergence logit matching | Built |
| Sleep replay / offline consolidation | Priority-based replay from core-set | Built |

### Key Neuroscience Findings Incorporated

1. **Consolidation is a staged cascade (Nature 2025):** Long-term memory is shaped by a cascade of molecular "timers" unfolding across brain regions, with the thalamus as a decision hub. This is the biological basis for our two-stage gate: cheap early filters (thalamic/hippocampal gating) before expensive consolidation (cortical protein synthesis analogue).

2. **Recall destabilizes memories (Nature 2025-2026):** When a consolidated memory is recalled, the hippocampus re-engages, making the memory editable again. This requires fresh resources (neurogenesis) and causes precision loss in nearby memories. This is the basis for our recall-destabilization: retrieval opens a labile window, restabilization requires fresh importance computation.

3. **Engram sparsity and competition:** Neuronal competition shapes which cells form an engram. This is the basis for our k-WTA sparse activation and the three-path fast system (competition between general/specific/residual pathways).

4. **Hippocampal indexing:** The hippocampus maintains a sparse index that reactivates cortical patterns, not the full memory content. This is the basis for functional addressing (probe sets instead of weight masks).

5. **Metaplasticity / Bayesian synaptic uncertainty:** Synapses don't just hold values — they hold plasticity states tracking how settled their importance is. This is the basis for our stability-weighted decay in fast paths.

---

## 3. Architecture Components — Detailed

### 3.1 Base Model (model.py)

**Two options:**
- `SlowCNN`: 2-conv CNN (3.2M params). For MNIST-scale tasks (28×28 grayscale).
- `BiDirSlowResNet`: ResNet-18 backbone (~17M params). For CIFAR-scale tasks (32×32 RGB).

**ResNet Split Architecture:**
```
Input (3×32×32)
  → Shared: conv1→bn1→relu→layer1→layer2 (feature extraction)
  → Slow path: layer3→layer4→avgpool→fc (consolidated knowledge)
  → Fast paths (3): parallel from shared features
  → Output: slow_out + gen_out + spec_out + resid_out
```

**Bi-directional confidence modulation:**
- Slow layer produces softmax confidence ∈ [0, 1]
- Fast effective LR = `base_lr × (1 - confidence)`
- When slow is confident, fast learns slowly (don't override)
- When slow is uncertain, fast learns quickly (adapt)

### 3.2 Fast Paths (model.py — FastDecayPath)

Three parallel fast-weight pathways with different decay rates:

| Path | Hidden Units | Decay Rate | Half-life | Purpose |
|------|-------------|------------|-----------|---------|
| General (gen) | 128 | 0.005 | ~5 epochs | Broad recurring patterns |
| Specific (spec) | 32 | 0.05 | ~0.5 epoch | Episode-level details |
| Residual (resid) | 16 | 0.2 | ~0.25 epoch | Edge cases, error correction |

**Usage-weighted decay (metaplasticity):**
- Each hidden unit has a continuous stability score tracking mean activation magnitude
- `stability += |activation|.mean(dim=0)` — accumulated, never decreases
- `decay_factor = decay_rate / (1 + stability × 0.01)` — higher stability = slower decay
- `weight.lerp(initial_weight, decay_factor)` — weights decay toward initialization, not zero

This means: frequently active units retain their weights (they're load-bearing). Rarely active units decay quickly (they're not contributing).

**Maps to brain:** Metaplasticity — the brain maintains synapses proportional to their contribution history. Units that are consistently active are "potentiated" — harder to change.

### 3.3 Episodic Buffer (buffer.py)

**BufferEntry stores:**
- `inputs`, `labels`, `logits`: raw experience data
- `features`: cached fc1/ResNet activations for retrieval
- `core_inputs`, `core_labels`, `core_logits`, `core_features`: protected subset (survives FIFO eviction)
- `pred_error_history`: prediction error over time (for Stage 1 gate)
- `core_acc_history`: core-set accuracy over time (for destabilization detection)
- `commit_accuracy`: accuracy at commit time (baseline for relative drift detection)
- `destabilize_count`, `last_destabilized_step`: cooldown tracking

**Clustering:** Currently by class label (0-9 for digit tasks, 0-9 for CIFAR). This is a deliberate simplification — sidesteps the "what counts as one belief" question. For the original vision (3D exploration), clustering should be emergent from the data (e.g., by feature-space proximity, not by label).

**Feature caching:** After `recompute_errors()`, features are cached via `cache_features(feature_fn)`. The feature_fn extracts fc1 or shared ResNet activations. This enables fast similarity-based retrieval.

**Retrieval:** `retrieve(query_features, k, device)` — finds top-K most similar stored examples by cosine similarity in feature space. Used for episodic retrieval at inference time.

### 3.4 Promotion Gate (gate.py)

**Stage 1 — Cheap filter (is_candidate):**

```python
def is_candidate(entry, freq_threshold=500, persist_window=10):
    frequent = entry.seen_count >= freq_threshold
    persistently_surprising = not is_decreasing(recent_errors)
    return frequent or persistently_surprising
```

Two independent signals trigger candidacy (OR, not AND):
- **Frequency:** "This keeps happening" — a regularity worth encoding
- **Persistent surprise:** "I keep getting this wrong despite repeated exposure"
- `is_decreasing` fits a linear trend on recent error history; flat-or-rising = persistent surprise

**CURRENTLY UNTESTED:** On MNIST and CIFAR benchmarks, every class accumulates enough samples within 1 epoch to exceed `freq_threshold`. The filter has never actually rejected any class. Its value would only show in environments with noise, rare classes, or redundant experiences.

**Maps to brain:** The thalamocortical cascade that acts as a molecular "timer" — cheap early checkpoints gate access to expensive consolidation. The two signals (frequency and persistent surprise) correspond to different routes in this cascade.

**Stage 2 — Expensive validation (validate_and_commit):**

```python
1. Subsample candidate data to 256 images (for efficiency)
2. Measure old_error on candidate, old_replay_loss on replay sample
3. Create shadow model copy
4. Fine-tune shadow on candidate + replay data (5 steps)
5. Measure new_error, new_replay_loss
6. If gain > eps_gain AND forgetting < eps_forget:
   - Copy shadow weights to model
   - Compute Fisher diagonal (vestigial)
   - Store importance mask and theta_star
   - Return True
7. Otherwise return False
```

Key design choice: **data subsampling to 256 images.** This makes the shadow fine-tune tractable (1.9s vs 19.8s for 5000 images). The validation only needs to detect DIRECTION of improvement — it doesn't need full convergence.

**Efficiency fix applied:** The joint loader now uses subsampled candidate data (not all 5000+ images). This was the #1 bottleneck — without it, each commit took 20s on CIFAR with ResNet.

### 3.5 Recall-Destabilization (gate.py)

**Three-criteria detection (check_destabilization_v3):**

All three must fire for destabilization to trigger:

1. **Prediction error (relative):** `current_core_acc < commit_accuracy - margin`
   Not an absolute threshold — compares against cluster's own baseline at commit time.
2. **Novelty:** `cosine_distance(core_feature_centroid, recent_data_feature_centroid) > threshold`
   Checks if new buffer data is genuinely different from the core-set (true drift, not retrieval failure).
3. **Persistence:** Low accuracy holds for N=3 consecutive checks.
   Prevents single-batch fluctuations from triggering destabilization.

**Partial degradation (destabilize_partial):**

When destabilization is triggered:
1. **Degrade old trace (strength-scaled):** `importance_mask *= (1 - strength)` where `strength = min(1.0, acc_drop / 0.50)`. Strong contradiction → near-complete degradation. Weak contradiction → partial retention.
2. **Fine-tune on current data** (contradictory examples, no replay for this cluster).
3. **Check collateral damage** — fine-tune is rejected if other clusters' accuracy drops.
4. **Restabilize:** Update core-set, recompute commit accuracy, reset history.

**Maps to brain:** The ubiquitin-proteasome system (UPS) — when a memory is retrieved and contradicted, old synaptic proteins are tagged for degradation before new ones are synthesized. The degradation strength is proportional to the prediction error magnitude.

**CURRENT STATUS:** Prototyped but not fully validated. The false-positive issue from v1 (firing on poorly-consolidated clusters) is addressed by the relative accuracy check, but hasn't been tested because no current benchmark has concept drift.

### 3.6 Priority Replay (train.py — build_replay_sample)

Instead of uniform per-cluster sampling, replay samples are weighted by:

```python
priority = max(0.01, acc_drop × 3.0 + pred_err × 2.0)
```

Where `acc_drop = commit_accuracy - current_core_accuracy` (how much this cluster is being forgotten) and `pred_err` = most recent prediction error (how confused the model is by this cluster).

**Effect:** Clusters that are rapidly being forgotten get 3× more replay bandwidth than stable ones.

**CURRENT STATUS:** Mathematically correct but marginal effect on current benchmarks. All clusters degrade at similar rates on Split-MNIST and CIFAR, so priorities are similar. Would show value in environments with ASYMMETRIC forgetting (some concepts are harder to retain than others).

### 3.7 Replay Weight Auto-Scaling (train.py)

```python
replay_mult = replay_weight × (1.0 + 0.15 × len(committed_clusters))
```

As more knowledge accumulates, the replay pressure increases proportionally. At replay_weight=2.5:
- 0 committed clusters: 2.5× (but no effect — nothing to replay)
- 5 committed clusters: 2.5 × 1.75 = 4.4×
- 10 committed clusters: 2.5 × 2.5 = 6.25×

**Why no cap:** The user explicitly rejected capping. The reasoning is that replay weight should auto-scale with the accumulated knowledge. If the model knows 100 clusters, it needs strong replay reinforcement to maintain them all.

### 3.8 Logit Distillation (train.py + buffer.py)

When logits are available in the buffer (stored at add-time), replay uses KL divergence between current logits and stored logits, not cross-entropy with hard labels:

```python
log_probs = log_softmax(current_logits)
target_probs = softmax(stored_logits)
replay_loss = KL_div(log_probs, target_probs)
```

**Why KL instead of MSE:** KL divergence is scale-invariant and directly measures distribution divergence. It preserves relationships between classes (e.g., "this could be a 0 but also looks somewhat like a 6").

**CURRENT STATUS:** Infrastructure works but doesn't help on Split-MNIST (classes are unambiguous — soft targets carry no extra information). Would show value on data with ambiguous class boundaries (CIFAR, natural images, Blurry MNIST).

### 3.9 Episodic Retrieval (model.py + buffer.py)

At inference time, when the model's confidence is low (< 0.85):

1. Extract fc1 features from current input
2. Query buffer for top-K similar stored examples (by cosine similarity)
3. Re-run retrieved examples through fast paths
4. Add retrieved outputs as bias to combined output

```python
episodic_bias = mean(fast_paths(retrieved_example) for retrieved_example in top_k)
output = slow_out + fast_out + episodic_bias
```

**Maps to brain:** Hippocampal pattern completion — a partial cue (current input) triggers retrieval of similar past experiences, and those retrieved patterns bias current processing.

**CURRENT STATUS:** Requires temporal continuity in the data stream (consecutive frames sharing visual features). On static image benchmarks, retrieved examples are too different from current input to be useful. The drift stream benchmark (oscillating blur/color/rotation) provides this temporal continuity.

---

## 4. Benchmarks Implemented

### Split-MNIST (primary benchmark)
- 5 tasks, 2 classes each (0/1, 2/3, 4/5, 6/7, 8/9)
- Class-incremental (10-class single-head output, no task labels at test)
- Current best: **79.3% ACC, BWT -0.139** (v3, 5 seeds)
- SOTA comparison: DER/DER++ ~85-92%

**What it tests:** Basic consolidation and forgetting prevention under clean task boundaries.

**What it DOESN'T test:** Temporal continuity, concept drift, online learning, noisy data, class imbalance, generalization.

### Permuted MNIST
- 10 tasks, same 10 digit classes each with different pixel permutation
- Domain-incremental (task = transform, classes stay the same)
- Current result: ~76% ACC, BWT -0.11 (v3 with episodic retrieval)

**What it tests:** Adaptation to changing input distributions. Cross-task invariance learning.

### Split-CIFAR-10 (harder benchmark)
- 5 tasks, 2 classes each (airplane/automobile, bird/cat, deer/dog, frog/horse, ship/truck)
- Class-incremental, same structure as Split-MNIST but with natural images
- Uses ResNet-18 backbone (17M params)
- Current result: ~90% on task 0, ~21% after task 4 (1 epoch)
- SOTA comparison: ResNet-based continual learning methods achieve ~50-85%

**What it tests:** Whether the architecture scales to harder visual data (requires deeper backbone, more epochs).

**Current issue:** 1 epoch per task is insufficient for CIFAR convergence. The model doesn't learn good features before the next task arrives.

### Drift Stream (custom benchmark)
- Continuous stream of MNIST or CIFAR images with smoothly varying Gaussian blur
- Temporal continuity: consecutive frames have almost identical blur
- Tests: episodic retrieval, logit distillation, priority replay, online adaptation
- No task boundaries, no epochs — true streaming evaluation

**What it tests:** The conditions under which our advanced features (episodic retrieval, priority replay, logit distillation) should finally show their value.

---

## 5. Current State Assessment

### What Actually Works (empirically validated)

| Feature | Evidence | Confidence |
|---------|----------|------------|
| Three-path fast/slow with decay | 6× variance reduction, +3% ACC over single-path | High |
| Replay weight auto-scaling | +11.6% ACC at replay_weight=2.5 | High |
| Replay-based forgetting prevention | 79.3% ACC vs 19% naive | High |
| ResNet backbone on CIFAR | Forward pass works, 90% on task 0 | Medium |
| Subsampled validate_and_commit | 10× speedup (19.8s → 1.9s per commit) | High |

### What's Built but Not Proven

| Feature | Why Not Proven | What Would Prove It |
|---------|----------------|-------------------|
| Stage 1 gate filter | All classes exceed freq_threshold in 1 epoch | Noisy environment with rare/redundant classes |
| Priority replay | All clusters degrade similarly on current benchmarks | Asymmetric forgetting environment |
| Logit distillation | Unambiguous classes on MNIST | Ambiguous classes / overcomplete data |
| Episodic retrieval | Confidence rarely below threshold on static images | Drift stream where blur makes confident classification impossible |
| Destabilization v3 | No concept drift in current benchmarks | Environment where labels change over time |
| CIFAR convergence | 1 epoch too few for ResNet | 10+ epochs per task or pretrained backbone |
| Fast paths as one-shot learner | Fast learning via gradient steps, not true one-shot | Memory retrieval at inference (episodic retrieval) |

### What's Missing Entirely

| Component | Priority | Why |
|-----------|----------|-----|
| World priors (physics rules) | Critical | Without priors, the model relearns basic features every time |
| Active exploration environment | Critical | Without it, most advanced features (episodic retrieval, destabilization) cannot be tested |
| Intrinsic motivation (curiosity) | Critical | Original vision depends on this; deferred since v1 |
| Online streaming mode | High | Current epoch-based training doesn't match real-time learning |
| Deliberate forgetting | Medium | The field now recognizes this as necessary for privacy and adaptability |
| Pretrained backbone | Medium | Would solve the CIFAR convergence problem immediately |

---

## 6. Key Decisions and Why

### D1: Replay carries the protection load, not Fisher masks
**Finding:** The per-layer freeze diagnostic proved that diagonal Fisher masks structurally cannot protect distributed representations at this scale. 25% per-layer frozen → 0% task retention.

**Implication:** The Fisher mask in Stage 2 is vestigial. It provides weak directional bias but can't prevent forgetting. The actual protection comes from interleaved replay during training.

**Future consideration:** If a better importance metric is found (e.g., SI-based trajectory importance instead of post-hoc Fisher), it could replace the current Fisher mask. But replay will remain the primary mechanism.

### D2: Three fast paths, not one
**Finding:** A single fast path with uniform decay performs well but has higher variance. Three paths with different decay rates smooth the learning signal.

**Implication:** Different types of "fast" knowledge exist — broad patterns that should persist (gen), ephemeral details (spec), and edge cases (resid). Each needs a different timescale.

### D3: Confidence-modulated learning, not hard gating
**Design choice:** The slow layer's confidence gates the fast learning rate continuously (not a binary on/off). This matches how the brain's neuromodulatory systems work — graded, not binary.

**Why not hard gating:** When the slow layer is moderately confident (0.7), the fast layer should learn moderately slowly (0.3 × base_lr), not flip between "full speed" and "stopped."

### D4: Functional addressing (probe sets), not weight masks
**Decision after Phase 3:** Belief addressing for destabilization uses probe sets (representative examples from the cluster's core-set), not Fisher-weight masks.

**Why:** Weight masks are distributed across millions of parameters and can't cleanly isolate one belief. Probe sets are concrete, interpretable, and naturally capture what makes a cluster distinct.

### D5: Per-class clustering is a placeholder
**Current:** Cluster ID = class label (0-9 for digits). This avoids the "what counts as one belief" question for now.

**Future:** Clustering should be emergent from the data — by feature-space proximity, not by label. This is required for the original 3D exploration vision where there are no predefined classes.

### D6: No cap on replay weight multiplier
**User decision:** The replay weight multiplier `rw × (1 + 0.15 × n_clusters)` has no upper bound. The logic: more accumulated knowledge requires proportionally more replay protection. For 100 clusters, this would be ~16×. Whether this causes plasticity problems at scale is unknown.

---

## 7. Efficiency Considerations

### Current Performance Profile (v6.0, ResNet on CIFAR)

| Operation | Time | Notes |
|-----------|------|-------|
| One epoch training (2 class CIFAR, ResNet) | ~7.3s | Forward + backward + step |
| validate_and_commit (subsampled 256 images) | ~1.9s | Was 19.8s without subsampling |
| recompute_errors (full 50K buffer) | ~4.3s | Scales linearly with buffer size |
| Full 5-task CIFAR benchmark | ~150s | 1 epoch per task, visible progress |

### Known Inefficiencies

1. **`recompute_errors` processes ALL buffer entries every epoch.** For 10 clusters with 5000 images each, that's 50K forward passes through the ResNet. Most of these are never used — only committed clusters need error recomputation.

2. **Buffer stores ALL inputs for ALL clusters.** With CIFAR-sized images, 50000 × 3 × 32 × 32 × 4 bytes = 576MB. For a streaming environment over months, this would scale to terabytes. The buffer needs a bounded-memory policy (reservoir sampling or core-set-only storage).

3. **`validate_and_commit` creates a deepcopy of the entire model.** While fast (0.0s for lazy copy), the shadow model still occupies GPU memory. For very large backbones, this might cause OOM.

4. **Episodic retrieval at inference is expensive.** Each retrieval computes cosine similarity against ALL buffer entries. For 50K entries, this is ~50000 × 256-dimensional dot products. Acceptable for batch inference but not for real-time single-sample inference.

---

## 8. Connection to the Original Vision

The original conversation (Claude-Adaptive AI model with continuous learning in simulated worlds.md) envisioned:

> "A model that grows all the time just like humans, with a world around it."

The architecture we've built is the MEMORY SYSTEM for this vision — but not the WORLD or the EXPLORATION DRIVE.

**What we have:**
- A way to store experiences (buffer with core-set protection)
- A way to consolidate important experiences into permanent knowledge (gate + fast/slow weights)
- A way to retrieve past experiences at inference (episodic retrieval)
- A way to update or correct committed beliefs (destabilization)
- A way to prioritize what to replay (priority replay)

**What we still need:**
- A WORLD to experience (simulated environment with physics, objects, space)
- A DRIVE to explore (curiosity reward / uncertainty reduction)
- A BODY to act (sensorimotor loop — observe → decide → act → observe)

The relationship:
```
World (environment) 
  → Agent observes (sensory input)
    → Fast system adapts immediately (fast-weight paths)
      → Buffer stores experience (episodic buffer)
        → Gate promotes important experiences (promotion gate)
          → Slow system consolidates (ResNet layers)
            → Agent uses knowledge to act
              → World changes
                → Loop continues
```

Destabilization fits here:
```
  → Agent acts based on committed knowledge
    → World contradicts expectation (prediction error)
      → Destabilization fires (3-criteria check)
        → Old belief is partially degraded
          → New evidence is integrated
            → Belief is restabilized
```

Intrinsic motivation fits here:
```
  → Agent's prediction error measured
    → Reduction in error over time = reward (curiosity)
      → Agent seeks experiences that reduce uncertainty
        → Agent explores novel but learnable situations
```

---

## 9. Basic Rules for Future Development

1. **No transformers.** Self-attention is explicitly avoided due to cost. The architecture uses CNNs and small fast-weight adapters.

2. **No foundation models.** No frozen pretrained weights for knowledge that should grow. "Frozen physics" priors (hardcoded rules) are acceptable.

3. **Efficiency matters.** Not just fast training, but no wasted computation on redundant operations. The architecture should not compute the same thing twice.

4. **Epistemic honesty.** If a mechanism doesn't work, document why. Don't keep it because it was expensive to build.

5. **Every mechanism maps to the brain.** Before adding a new component, check if it has a neuroscience basis. If it doesn't, question whether it's necessary.

6. **Test in the environment, not on the benchmark.** The benchmark is a sanity check. The true test is whether the architecture can learn continuously in a changing environment.

7. **Sequential validation.** Build one mechanism at a time, validate it, then add the next. Never build two untested mechanisms at the same time.

---

## 10. File Map

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `model.py` | Neural network architectures | `BiDirSlowResNet`, `FastDecayPath`, `SlowCNN`, `create_model` |
| `buffer.py` | Episodic memory with core-set | `BufferEntry`, `EpisodicBuffer` |
| `gate.py` | Promotion gate + destabilization | `is_candidate`, `validate_and_commit`, `check_destabilization_v3`, `destabilize_partial` |
| `train.py` | Training loops + experiment harness | `train_two_stage_gate`, `build_replay_sample`, `run_experiment`, `main` |
| `baselines.py` | Naive + EWC baselines | `train_naive`, `train_ewc`, `compute_fisher_diag`, `ewc_penalty` |
| `data.py` | Benchmark data generators | `get_split_mnist_tasks`, `get_permuted_mnist_tasks`, `generate_drift_stream`, `get_split_cifar10_tasks`, `generate_cifar_drift_stream` |
| `metrics.py` | Evaluation | `evaluate`, `compute_acc`, `compute_bwt` |
| `diagnose_fisher.py` | Fisher ranking diagnostic | — |
| `diagnose_freeze.py` | Per-layer freeze diagnostic | — |
| `test_destabilization.py` | Drift + destabilization test | — |

---

## 11. Running the Code

### Setup
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision scipy matplotlib numpy
```

### Quick test (gate on Split-MNIST)
```bash
python3 train.py --configs two_stage_gate --seeds 42 --epochs 1 --use-fast-layer
```

### Full benchmark (5 seeds, all configs)
```bash
python3 train.py \
    --configs naive ewc two_stage_gate \
    --seeds 42 43 44 45 46 \
    --epochs 1 \
    --use-fast-layer \
    --replay-weight 2.5 \
    --freq-threshold 500 \
    --eps-gain 0.001 \
    --eps-forget 0.50 \
    --core-size 100 \
    --k-wta 64
```

### CIFAR benchmark (ResNet, 5 epochs)
```bash
python3 train.py \
    --configs two_stage_gate \
    --seeds 42 \
    --epochs 5 \
    --use-fast-layer \
    --benchmark split_cifar10 \
    --replay-weight 2.5
```

### Drift stream (temporal continuity for episodic retrieval)
```bash
python3 train.py \
    --configs two_stage_gate \
    --seeds 42 \
    --epochs 1 \
    --use-fast-layer \
    --benchmark drift_stream \
    --drift-frames 5000 \
    --episodic-retrieval-k 5
```

### Destabilization test (requires drift)
```bash
python3 test_destabilization.py
```

---

## 12. Handoff to Next Agent

### Current State (commit e5a8a77)

All v1.1 through v6.0 changes are committed. The architecture is stable and testable. The ResNet backbone is functional but needs more epochs per task for CIFAR.

### Most Important Open Questions

1. **Does the Stage 1 gate actually provide value?** It has never filtered anything because all benchmarks provide abundant examples of each class. A test with rare/noisy classes would answer this.

2. **Does episodic retrieval help under temporal continuity?** The drift stream benchmark is built for this but hasn't been run with episodic retrieval enabled and properly tuned.

3. **Does destabilization work under real concept drift?** The three-criteria detection is designed to avoid false positives but hasn't been tested because no benchmark has natural drift.

4. **Can the architecture scale to streaming (single-pass) learning?** Current training uses epochs and task boundaries. True online processing would be required for the original vision.

### Most Urgent Next Steps

1. **Pretrained ResNet backbone** — Load ImageNet weights for the shared and slow-path layers. This would immediately improve CIFAR convergence.

2. **Drift stream optimization** — Tune episodic retrieval, priority replay, and logit distillation specifically for the continuous drift setting.

3. **World environment** — Implement a 2D grid world (MiniGrid or similar) with physics and objects. This is where the architecture's features can finally be tested in their intended context.

4. **Intrinsic motivation** — Implement the uncertainty-reduction curiosity reward. This is the part of the original vision that has been deferred the longest.

---
