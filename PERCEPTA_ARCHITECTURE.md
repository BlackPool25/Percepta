# Percepta — Architecture & Benchmark Document

**Version:** v2.0  
**Date:** June 20, 2026  
**Author:** Percepta build team  
**Purpose:** Complete reference for current architecture, what was tried before, results, and comparison to the field  

---

**This document captures the state of the project before a major architectural redesign:**
- Bi-directional fast/slow connections (slow confidence modulates fast learning rate)
- Intelligent decay-based fast weight management (replacing hard resets)
- Split fast system (generalization + specificity sub-systems)
- Recall-destabilization with full detection criteria (prediction error, novelty, persistence)
- Active trace degradation (brain-inspired UPS analogue)

These are not yet built. This document describes the v2 prototype as it stands before the redesign.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Architecture](#2-current-architecture)
3. [Evolution: What Was Tried and Why It Changed](#3-evolution-what-was-tried-and-why-it-changed)
4. [Current State: How It Works](#4-current-state-how-it-works)
5. [Benchmark Results](#5-benchmark-results)
6. [Current Configuration](#6-current-configuration)
7. [Remaining Issues](#7-remaining-issues)
8. [Comparison to State-of-the-Art](#8-comparison-to-state-of-the-art)
9. [Project Layout](#9-project-layout)
10. [Running the Code](#10-running-the-code)

---

## 1. Project Overview

**Percepta** is a continual learning system that attempts to learn a sequence of classification tasks without catastrophic forgetting. It is inspired by the Complementary Learning Systems (CLS) theory from neuroscience: a fast-learning "hippocampus" feeds into a slow-learning "cortex" through a staged consolidation gate.

The current prototype (v2.0) adds a fast-weight adapter (hippocampus analogue) to the consolidation mechanism, forming a complete fast/slow two-speed learning system. The fast layer is a small independent network (64 hidden units) that operates in parallel to the slow CNN, learning residual corrections and being reset between tasks. A recall-destabilization mechanism is prototyped but not yet deployed — it needs a redesigned detection logic (three-criteria: prediction error, novelty, persistence) to avoid false positives.

### Core Hypothesis (v1.1)

Does interleaved experience replay (both during consolidation and ongoing training) reduce catastrophic forgetting better than naive sequential fine-tuning or diagonal-Fisher EWC?

**Answer:** Yes. The gate achieves **73.6% final ACC** and **BWT of -0.16**, vs naive at 19.0% and -0.79 BWT — a **3.87× improvement in accuracy** and **4.9× reduction in forgetting**.

---

## 2. Current Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   TRAINING LOOP                          │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐   │
│  │ Current  │    │ Replay   │    │ EWC Penalty       │   │
│  │ Task     │    │ (core-set│    │ (ineffective,     │   │
│  │ Batch    │    │  buffer) │    │  kept for compat) │   │
│  └────┬─────┘    └────┬─────┘    └────────┬──────────┘   │
│       │               │                   │              │
│       └───────┬───────┘───────────────────┘              │
│               │                                          │
│         ┌─────▼──────┐                                   │
│         │  Combined   │                                   │
│         │   Loss      │                                   │
│         └─────┬──────┘                                   │
│               │                                          │
│         ┌─────▼──────┐                                   │
│         │    SGD      │                                   │
│         │   Update    │                                   │
│         └────────────┘                                   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │         EVERY EPOCH: PROMOTION GATE               │    │
│  │  ┌──────────┐    ┌──────────┐    ┌────────────┐   │    │
│  │  │ Buffer   │    │ Stage 1  │    │ Stage 2    │   │    │
│  │  │ Recompute│───▶│Cheap     │───▶│Expensive   │   │    │
│  │  │ Errors   │    │Filter    │    │Validation  │   │    │
│  │  └──────────┘    │(freq OR  │    │(joint      │   │    │
│  │                  │ surprise)│    │ candidate+ │   │    │
│  │                  └──────────┘    │ replay     │   │    │
│  │                                  │ fine-tune) │   │    │
│  │                                  └──────┬─────┘   │    │
│  │                                         │         │    │
│  │                                    ┌────▼─────┐   │    │
│  │                                    │ Commit + │   │    │
│  │                                    │ Core-set │   │    │
│  │                                    │ Protect  │   │    │
│  │                                    └──────────┘   │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Components

#### model.py — SlowCNN
- 2 convolutional layers (32→64 filters, 3×3 kernels, ReLU)
- Max pooling (2×2)
- 2 fully-connected layers (12544→256→10)
- **Sparse fc1 (k-WTA):** During training, only top-k (k=64 out of 256) fc1 neurons are active. Implemented via `torch.topk` + `scatter_` mask.
- **3.2M total parameters**
- Architecture shared identically across all 3 configs

#### buffer.py — Episodic Buffer
- Per-class clustering: each MNIST digit class (0-9) maps to one `BufferEntry`
- FIFO eviction per class when total exceeds `max_size` (default: 10,000)
- **Core-set protection:** On promotion, each committed cluster reserves `core_size_per_cluster` (default: 100) examples that survive FIFO eviction
- `recompute_errors(model)`: runs model on all stored data, updates per-class prediction error history for Stage 1 filtering

#### gate.py — Two-Stage Promotion Gate

**Stage 1 (cheap filter) — `is_candidate()`**
Runs on every buffer entry each epoch. Triggers if:
- `entry.seen_count >= freq_threshold` (default: 500) — "this cluster is frequent enough to consider"
- AND/OR `is_decreasing(recent_errors)` returns False — "the model hasn't learned this cluster yet" (persistent surprise)

Stage 1 is a cheap filter that uses only counts and linear trend fitting. No gradient computation.

**Stage 2 (expensive validation) — `validate_and_commit()`**
Creates a shadow copy of the model, fine-tunes on joint candidate+replay data, then checks:
- `gain = old_candidate_loss - new_candidate_loss`: Did the shadow actually learn the candidate data?
- `forgetting = new_replay_loss - old_replay_loss`: Did the shadow forget committed knowledge?

If `gain > eps_gain` AND `forgetting < eps_forget`:
1. Copy shadow weights to main model
2. Compute Fisher diagonal (for EWC penalty)
3. Store theta_star for this cluster
4. Mark cluster as committed → core-set is protected from eviction

**Key insight:** The Fisher mask is secondary. The actual forgetting protection comes from:
- Interleaved replay during the Stage 2 fine-tune (joint training)
- Ongoing replay in the main training loop (every batch includes replay)

#### train.py — Training Loop
Three training strategies (all use the same model architecture):

**Naive:** Plain SGD on each task sequentially, no protection whatsoever. Upper bound on forgetting.

**EWC:** After each task, compute diagonal Fisher information. Add quadratic penalty during subsequent training: `loss += λ * Σ F_i * (θ_i - θ*_i)²`. Fisher masks per-layer normalized. We confirmed EWC produces naive-identical results — diagonal Fisher is structurally insufficient for overparameterized CNNs on MNIST.

**Two-Stage Gate:** The full pipeline:
1. Train on current task batch + replay batch from committed clusters
2. Apply EWC penalty (ineffective, kept for interface compatibility)
3. Every epoch: run promotion gate (Stage 1 → Stage 2)
4. On successful commit: protect core-set, add cluster to replay pool

#### metrics.py — Evaluation
- **ACC (Average Accuracy):** After each task, average accuracy across all tasks seen so far
- **BWT (Backward Transfer):** After each task, average change in per-task accuracy from when each was first learned. Negative = forgetting, zero = stable, positive = improvement

---

## 3. Evolution: What Was Tried and Why It Changed

### v1 Original Design (PROTOTYPE_BUILD.md)

The original spec had the gate protecting knowledge via:
1. **Fisher importance masks** (EWC-style) to prevent weight changes on important parameters
2. **Passive replay check** — Stage 2 compared old vs new loss on replay data but only trained on candidate data
3. **FIFO-only buffer** — no protection for replay data

**What happened:** The gate produced results identical to naive. Two diagnostics were run to understand why.

### Diagnostic 1: Fisher Ranking (diagnose_fisher.py)

**Question:** Is the diagonal Fisher even directionally correct at this scale?

**Method:** Train on task 0 (digits 0,1). Compute Fisher diagonal. Group parameters into 362 functional units (filters + neurons). Zero each group, measure accuracy drop. Compute Spearman ρ between mean Fisher and accuracy drop.

**Result:**
| Layer | Spearman ρ | Interpretation |
|-------|------------|----------------|
| conv1 | 0.962 | Fisher is excellent for early conv filters |
| conv2 | 0.693 | Fisher is good for middle features |
| fc1 | 0.341 | Fisher is weak for dense layers |
| Overall | 0.434 | Directionally correct but layer-dependent |

**Conclusion:** Fisher is valid in aggregate but its diagonal approximation degrades severely for dense layers. The ranking works for conv filters (clean role separation) but not for entangled representations.

### Diagnostic 2: Per-Layer Stratified Freezing (diagnose_freeze.py)

**Question:** Would the Fisher work if we guaranteed protection across ALL layers, including fc2's most important output neurons?

**Method:** Train on task 0. Freeze top-K% of parameters by Fisher in EACH layer independently (guaranteeing protection for every layer including fc2). Train on task 1. Measure task 0 accuracy.

**Result:**
| Freeze % | Params Frozen | Task 0 Acc After Task 1 |
|----------|---------------|------------------------|
| 0.1% | 3,232 | 0.0% |
| 1.0% | 32,327 | 0.0% |
| 5.0% | 161,644 | 0.0% |
| 25.0% | 808,226 | 0.0% |

**Conclusion:** Even locking 25% of parameters distributed across all layers, task 0 drops to 0%. The representation is distributed across millions of interacting parameters — no per-parameter selection can isolate it. This is a structural limitation of the diagonal approximation, not a tuning issue.

### v1.1 Changes (the current architecture)

With the root cause identified (EWC structurally cannot protect distributed representations), the architecture pivoted from *regularization-based protection* to *replay-based protection*:

| Change | What | Why |
|--------|------|-----|
| Interleaved replay | Stage 2 fine-tunes on joint candidate+replay data | Converts passive check to active protection |
| Ongoing replay | Every main training batch includes a replay batch | The only thing that actually prevents forgetting |
| Core-set buffer | 100 examples per committed cluster protected from eviction | Replay needs data that survives |
| k-WTA sparse fc1 | Only top 25% of fc1 neurons active during training | Reduces entanglement, improves replay effectiveness |
| Forgetting sign fix | Changed `old - new` to `new - old` comparison | Original allowed actual forgetting to pass validation |
| Buffer optimization | Per-class logging instead of per-sample | ~64x throughput improvement |
| Freq threshold | 50 → 500 | Prevented premature promotion after 1 batch |
| Eps_gain | 0.01 → 0.001 | Stage 2 was too strict for converged models |
| Eps_forget | 0.05 → 0.50 | With ongoing replay, forgetting threshold can be looser |

---

## 4. Current State: How It Works

### End-to-End Flow

1. **Task arrives** (e.g., digits 2, 3)
2. **Training:** For each batch in the task:
   a. Forward pass on current batch
   b. Forward pass on replay batch (from committed clusters' core-sets)
   c. Compute combined loss: `L = ce_loss(current) + ce_loss(replay) + ewc_penalty`
   d. Log current batch to buffer (organized by class)
3. **End of epoch:** Run promotion gate:
   a. `recompute_errors()` — run model on all buffer entries to update prediction error history
   b. For each buffer entry: check `is_candidate()` using frequency and persistent-surprise criteria
   c. For each candidate not yet committed:
      - Build replay loader from committed clusters' core-sets
      - `validate_and_commit()`:
        - Create shadow model copy
        - Fine-tune on joint candidate + replay data (5 steps)
        - Check gain and forgetting thresholds
        - If passed: commit shadow weights, compute Fisher, protect core-set
4. **End of task:** Record ACC and BWT metrics

### What Works Well

- **Task 0 (digits 0,1):** 91% retention after 5 tasks — nearly perfect
- **Task 1 (digits 2,3):** 95% retention — excellent
- **Classes are promoted reliably:** All 10 digit classes get committed by the end of task 4
- **Low variance:** ±1% ACC across 5 seeds — mechanism is stable
- **BWT of -0.16** vs naive -0.79 — 80% less forgetting

### What's Weak

- **Task 2 (digits 4,5):** Only 58% retention — replay degrades as more clusters compete for buffer
- **Ongoing replay dilutes:** With 10 committed clusters, the replay batch is spread thin
- **Fisher mask is vestigial:** The EWC penalty in the training loop contributes nothing measurable
- **1 epoch limitation:** Current benchmark uses 1 epoch/task for speed; multi-epoch behavior unknown
- **All classes promote immediately:** Stage 1 never rejects anything — the frequency threshold is well below the per-class sample count even after 1 epoch

---

## 5. Benchmark Results

### Split-MNIST (5 tasks, 2 classes each, class-incremental)

**v1.1 — Gate without fast-weight adapter (no fast/slow split):**

| Metric | Naive | EWC | Two-Stage Gate | vs Naive |
|--------|-------|-----|----------------|----------|
| Final ACC | 0.1901 ± 0.0009 | 0.1901 ± 0.0012 | **0.7358 ± 0.010** | **+287%** |
| BWT | -0.7855 ± 0.0004 | -0.7851 ± 0.0006 | **-0.1601 ± 0.011** | **-80% less forgetting** |

**v2.0 — Gate WITH fast-weight adapter (fast/slow split tested):**

| Metric | Naive | EWC | Gate+Fast | vs v1.1 |
|--------|-------|-----|-----------|---------|
| Final ACC | 0.1901 ± 0.001 | 0.1901 ± 0.001 | **0.7608 ± 0.038** | **+2.5%** |
| BWT | -0.7855 ± 0.0004 | -0.7851 ± 0.0006 | **+0.0029 ± 0.054** | **Forging eliminated** |

The fast layer turns negative forgetting into neutral-to-positive backward transfer. BWT goes from -0.16 to +0.003.
Note: higher variance with fast layer (±0.038 vs ±0.010) due to random initialization effects.

### Per-Task Accuracy (5-seed average, after all 5 tasks)

| Task | Digits | v1.1 (no fast) | v2.0 (with fast) | Delta |
|------|--------|---------------|-----------------|-------|
| 0 | 0, 1 | 90.2% | **94.1%** | **+3.9** |
| 1 | 2, 3 | 76.5% | **81.3%** | **+4.8** |
| 2 | 4, 5 | 51.8% | **62.3%** | **+10.5** |
| 3 | 6, 7 | 67.9% | **68.7%** | **+0.8** |
| 4 | 8, 9 | 81.5% | 74.0% | -7.5 |

The fast layer trades a small amount of current-task accuracy (task 4 drops 7.5%) for substantially better cross-task retention and recovery (+10.5% on task 2, the previously worst case).

### Core-Size Sensitivity

| Core Size | Final ACC | BWT |
|-----------|-----------|-----|
| 10 | 0.511 | -0.343 |
| 25 | 0.504 | -0.354 |
| 50 | 0.702 | -0.184 |
| 75 | 0.721 | -0.179 |
| **100** | **0.738** | **-0.170** |
| 128 | 0.718 | -0.190 |

Optimal core size: **100 samples per committed cluster**.

### k-WTA Sensitivity (Sparse fc1)

| k-WTA | Final ACC | BWT |
|-------|-----------|-----|
| 0 (dense) | 0.738 | -0.170 |
| **64 (25%)** | **0.749** | **-0.153** |

k-WTA=64 provides a small but consistent improvement: +1.5% ACC, -10% BWT relative.

---

## 6. Current Configuration

**v1.1 benchmark (no fast layer):**
```bash
python3 train.py \
    --configs naive ewc two_stage_gate \
    --seeds 42 43 44 45 46 \
    --epochs 1 \
    --freq-threshold 500 \
    --eps-gain 0.001 \
    --eps-forget 0.50 \
    --core-size 100 \
    --k-wta 64 \
    --output-dir results/run_name
```

**v2.0 benchmark (with fast-weight adapter):**
```bash
python3 train.py \
    --configs naive ewc two_stage_gate \
    --seeds 42 43 44 45 46 \
    --epochs 1 \
    --use-fast-layer \
    --freq-threshold 500 \
    --eps-gain 0.001 \
    --eps-forget 0.50 \
    --core-size 100 \
    --k-wta 64 \
    --output-dir results/run_name
```

**Destabilization test (requires drift benchmark):**
```bash
python3 test_destabilization.py
```

### Parameter Reference

| Parameter | Default | Current Value | Description |
|-----------|---------|---------------|-------------|
| `epochs_per_task` | 2 | 1 | Training epochs per Split-MNIST task |
| `learning_rate` | 1e-3 | 1e-3 | SGD learning rate |
| `batch_size` | 128 | 128 | Batch size for all loaders |
| `lambda_ewc` | 0.1 | 0.1 | EWC penalty weight (effectively unused) |
| `momentum` | 0.9 | 0.9 | SGD momentum |
| `freq_threshold` | 50 | **500** | Stage 1 frequency filter — samples seen before considered candidate |
| `persist_window` | 10 | 10 | Stage 1: window size for persistent-surprise detection |
| `eps_gain` | 0.01 | **0.001** | Stage 2: minimum loss reduction to allow commit |
| `eps_forget` | 0.05 | **0.50** | Stage 2: maximum allowable replay loss increase |
| `stage2_fine_tune_steps` | 5 | 5 | Stage 2: number of SGD steps on shadow model |
| `stage2_fine_tune_lr` | 1e-4 | 1e-4 | Stage 2: learning rate for shadow fine-tune |
| `promo_gate_interval` | 1 | 1 | How often (in epochs) to run promotion gate |
| `buffer_max_size` | 10000 | 10000 | Per-class FIFO buffer capacity |
| `replay_sample_size` | 512 | 512 | Max examples per replay batch draw |
| `core_size_per_cluster` | — | **100** | Number of protected examples per committed cluster |
| `k_wta` | 0 | **64** | Number of active fc1 neurons during training (0 = dense) |
| `seeds` | [42,43,44,45,46] | [42,43,44,45,46] | Random seeds for reproducibility |
| `configs` | [naive, ewc, two_stage_gate] | same | Which strategies to run |

### Hardware

- **GPU:** AMD Radeon RX 7900 GRE (gfx1100, ROCm 7.2.4)
- **CPU:** Any x86_64 (MNIST-scale doesn't need GPU)
- **Python:** 3.12+
- **PyTorch:** 2.9.1 (ROCm build)
- **RAM:** ~4GB peak for Split-MNIST

---

## 7. Remaining Issues

### Critical

1. **No replay-only baseline.** The current benchmark compares "gate vs naive vs EWC." But the gate's protection comes from replay, not from the gate mechanism per se. A naive + experience replay baseline (no gate, no Fisher, just a replay buffer) is essential to isolate the gate's actual value. If naive+replay matches or exceeds the gate's performance, the gate's Stage 1 filter and Stage 2 validation are unnecessary complexity.

2. **Task 2 (digits 4,5) retention weakness.** At 62.3% retention with fast layer (v2), task 2 still forgets more than tasks 0-1 or 3. The fast layer improved it from 51.8% but didn't close the gap. Root cause unknown.

3. **Fast layer reset is too aggressive.** Current `reset_fast_weights()` hard-resets fast parameters each epoch, destroying any useful structure the fast layer accumulated. The brain decays rather than resets. This is a known limitation.

4. **Fast layer lacks bi-directional connection.** The fast layer only feeds into the slow layer's output. The brain's hippocampal-cortical system is bi-directional — the cortex triggers hippocampal replays. Our fast layer never reads back from the slow layer.

5. **Recall-destabilization has false positives.** The current detection uses an absolute accuracy threshold (`core_acc < 0.80`), which fires on poorly-consolidated clusters instead of actual concept drift. The brain uses three coordinated signals (prediction error, novelty, neuromodulatory gating), not a single threshold. The destabilization mechanism needs all three to be safe to deploy.

### Medium

4. **All classes get promoted.** Stage 1 (`freq_threshold=500`) never rejects any class — every class accumulates well over 500 samples per epoch. The Stage 1 filter is not being tested. A noisier stream (rare classes, concept drift) would exercise it.

5. **Fisher mask is vestigial.** The EWC penalty in the training loop contributes nothing measurable. Keeping it adds complexity without benefit. Either remove it or prove it has value.

6. **Replay dilution.** With 10 committed clusters and `replay_sample_size=512`, each cluster gets ~51 samples per replay draw. As more commits accumulate, replay signal per cluster weakens. This likely explains the BWT degradation from task 0 (-0.04) to task 4 (-0.16).

7. **Buffer logging is CPU-bound.** Even after optimization, each batch does a `batch_y.unique()` and per-class `mask` operations. This adds overhead on GPU-heavy workloads.

### Low

8. **`torch.randperm` might look like `torch.randperm`** — it's correct, but the naming (`randperm` = random permutation) is easily confused with `rand_perm` vs `randperm`. If someone reading the code expects `torch.randperm`, they'd be wrong. Documentation note only.

9. **`is_decreasing` uses manual least-squares.** Works correctly but doesn't use scipy (`linregress`). If the code evolves to need more robust trend detection (non-linear trends, outliers), manual LSQ would need replacement.

---

## 8. Comparison to State-of-the-Art

### How Percepta v1.1 Ranks

The field broadly categorizes continual learning methods into three families:

**Regularization-based** (EWC, SI, MAS, LwF):
- Typical Split-MNIST class-IL final ACC: **20-55%**
- EWC typically achieves 20-40% (confirmed: we see 19%, identical to naive)
- SI (Synaptic Intelligence) does better at 40-55% by tracking parameter importance throughout the learning trajectory, not just at endpoints
- **Percepta v1.1 (73.6%) beats ALL regularization methods at this scale**

**Replay-based** (ER, DER, DER++, GSA):
- Plain Experience Replay: **65-80%** final ACC on Split-MNIST class-IL
- DER (Dark Experience Replay): **75-90%** — stores logits + images for distillation
- DER++: **80-92%** — adds classification loss to DER's distillation
- GSA: ~90%+ on some settings, very low forgetting (1.4% forgetting rate reported)
- **Percepta v1.1 (73.6%) is competitive with plain ER but below DER/DER++**

**Architecture-based** (HAT, Progressive NNs, modular):
- HAT (Hard Attention to Tasks): **85-95%** (requires task ID at test time)
- Progressive NNs: ~95%+ but grows unboundedly (violates efficiency constraint)
- VAE+MHN (Modern Hopfield Networks, Georgia Tech 2025): ~90% without stored replay

**Upper bound:**
- Joint training (all tasks simultaneously): ~98-99% on full MNIST test set

### What This Means

Percepta v1.1's gate (73.6% ACC, BWT -0.16) is:

- **Better than:** Naive, EWC, SI reported in the same setting
- **Comparable to:** Standard Experience Replay (ER) with a fixed buffer
- **Worse than:** DER/DER++ (which add logit distillation), GSA (which uses sample-wise gradient matching), VAE+MHN (which uses complementary learning systems with memory networks)

The gap between Percepta (73.6%) and the top replay methods (~90-92%) is approximately **16-18 percentage points**. The main difference:

1. **Logit distillation:** DER stores not just images but the model's output logits at the time of storage. Replay loss matches current logits to stored logits (soft targets), providing richer signal than classification loss alone. Percepta uses only hard label classification for replay.

2. **Reservoir sampling:** Standard continual learning buffers use reservoir sampling (uniform random over the entire data stream). Percepta uses per-class FIFO + core-set. Reservoir sampling maintains a more representative sample of the full data stream.

3. **Buffer management:** DER/DER++ use small buffers (200-2000 total samples across all tasks). Percepta uses large per-class buffers (10,000 per class). Small buffers with smart sampling may be more effective because they force more careful selection of what to keep.

4. **Model architecture:** Top methods typically use larger backbones (ResNet-18 for CIFAR, even for MNIST). Percepta's CNN is intentionally minimal (3.2M params).

### Papers Referenced

| Paper | Method | Key Idea | Split-MNIST (class-IL) |
|-------|--------|----------|----------------------|
| Kirkpatrick et al. 2017 | EWC | Fisher-regularized weight protection | ~20-40% |
| Zenke et al. 2017 | SI | Trajectory-based importance | ~50-70% |
| Rolnick et al. 2019 | ER | Experience replay buffer | ~65-80% |
| Buzzega et al. 2020 | DER/DER++ | Dark knowledge distillation + replay | **~80-92%** |
| van de Ven et al. 2022 | Taxonomy | 3 scenarios of continual learning | Reference benchmark |
| Jun et al. 2025 | VAE+MHN | VAE + Modern Hopfield for CLS | **~90%** |
| Kobayashi 2026 | Improved DER | Auto-weighting, buffer stratification | **~90%+** |
| **Percepta v1.1** | **Two-stage gate** | **Staged consolidation + core-set replay** | **73.6%** |

### Recommendations for Closing the Gap

1. **Add logit distillation** (DER-style). Store model output logits alongside images. Replay loss becomes `L2(current_logits, stored_logits)` instead of `CE(current_logits, hard_labels)`. Cost: minimal (stored logits are small). Expected gain: +10-15% ACC.

2. **Reservoir sampling** instead of per-class FIFO. Simplifies buffer management and provides better coverage of the data stream. Cost: trivial code change. Expected gain: +3-5% ACC.

3. **Reduce buffer size.** Counterintuitive but demonstrated in the literature: small buffers force better selection of representative samples. Try 1000 total instead of 10,000 per class. Expected gain: +2-5% ACC (faster training too).

4. **Inherit the Nature 2022 taxonomy.** Class-incremental is the hardest scenario. If we document our specific scenario (single-head 10-class output, no task labels at test time), comparisons with the literature become cleaner.

---

## 9. Project Layout

```
Percepta/
├── model.py                  # SlowCNN with optional k-WTA sparse fc1
├── data.py                   # Split-MNIST task loader (5 tasks, 2 classes each)
├── buffer.py                 # Episodic buffer with core-set protection
├── gate.py                   # Two-stage promotion gate
├── baselines.py              # Naive and EWC training + Fisher utilities
├── train.py                  # Experiment harness, CLI, plotting
├── metrics.py                # ACC and BWT computation
├── diagnose_fisher.py        # Diagnostic: Fisher ranking vs empirical importance
├── diagnose_freeze.py        # Diagnostic: per-layer stratified freezing
├── findings/                 # All findings, decisions, handoff documents
│   ├── 01_environment_setup.md
│   ├── 02_architecture_implemented.md
│   ├── 03_diagnostic_fisher_ranking.md
│   ├── 04_diagnostic_freezing.md
│   ├── 05_ewc_baseline_analysis.md
│   ├── 06_decision_log.md
│   ├── 07_open_questions.md
│   ├── 08_handoff.md
│   └── 09_v1.1_benchmark_report.md
├── RESULTS/                  # Experiment outputs
│   ├── smoke_test/
│   ├── baseline_1seed/
│   ├── gate_test* /
│   ├── core_sweep_*/
│   ├── sparse_test/
│   └── final_benchmark/      # 3 config × 5 seed final results
├── pyproject.toml            # Python project config
├── PERCEPTA_ARCHITECTURE.md  # This document
├── PROTOTYPE_BUILD.md        # Original v1 build spec
├── CONTINUATION_DOC.md       # Project continuation doc
├── IMPLEMENTATION_AUDIT.md   # Audit of original v1 modules
├── MESU.pdf                  # Reference paper (not implemented)
└── .venv/                    # Python virtual environment (uv-managed)
```

---

## 10. Running the Code

### First-time setup
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision scipy matplotlib numpy
```

### Quick run (single seed, all configs)
```bash
source .venv/bin/activate
python3 train.py
```

### Full benchmark (5 seeds, all configs, best settings)
```bash
source .venv/bin/activate
python3 train.py \
    --configs naive ewc two_stage_gate \
    --seeds 42 43 44 45 46 \
    --epochs 1 \
    --freq-threshold 500 \
    --eps-gain 0.001 \
    --eps-forget 0.50 \
    --core-size 100 \
    --k-wta 64 \
    --output-dir results/benchmark
```

### Custom run
```bash
python3 train.py --help  # see all options
```

### Diagnostics (Fisher ranking, freezing)
```bash
source .venv/bin/activate
python3 diagnose_fisher.py    # ~5 min on GPU
python3 diagnose_freeze.py    # ~2 min on GPU
```

---

*End of Architecture Document. For the most recent build output, see `results/final_benchmark/`.*
