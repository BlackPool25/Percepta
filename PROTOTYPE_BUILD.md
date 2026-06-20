# Percepta — Prototype v1 Build Spec

**What we're building:** A minimal test of the consolidation mechanism alone — a small CNN learning Split-MNIST sequentially, with an episodic buffer and a two-stage promotion gate controlling what gets locked into protected weights.

**Hypothesis to test:** Does the two-stage gate (cheap frequency/persistent-surprise filter → Fisher-validated integration) reduce catastrophic forgetting better than (a) naive sequential fine-tuning, and (b) plain single-mask EWC?

**Deferred (not in v1):** curiosity reward, world model, fast-weight layer, recall-destabilization, 3D environment, RL loop.

---

## 1. System Architecture — How It Works

### The Core Idea

The brain has two complementary learning systems: the **hippocampus** learns quickly and sparsely (episodic memories on the fly), and the **cortex** learns slowly and densely (generalized knowledge that lasts a lifetime). New experiences are first stored in the hippocampus, then gradually consolidated into the cortex over time — especially during sleep. We replicate this split in the AI: a cheap **fast layer** (hippocampus analogue) adapts moment-to-moment to whatever the agent is experiencing right now, while a protected **slow layer** (cortex analogue) accumulates corroborated, cross-checked knowledge that changes rarely and only when it's earned. Between them is a **promotion gate** that decides what's worth the expensive operation of modifying the slow layer's protected weights.

### v1 Builds Only the Slow-Layer Side

v1 strips this down to the minimum testable piece: **the promotion gate + slow layer alone**, without the fast layer. The model is a single small CNN — this is the "cortex." The episodic buffer is the "hippocampus" stand-in. The gate decides what transfers from buffer to protected weights.

### What Each Component Does

**Slow layer (model.py)** — The neural network that holds accumulated, consolidated knowledge. Its weights are protected by importance masks (Fisher information). Updates are rare and gated. In the brain, this is the cortex — dense, slow-changing, generalized representations.

**Episodic buffer (buffer.py)** — A short-term store of raw experiences. Each experience is logged with metadata: what was seen, what was predicted, how wrong the prediction was, how many times it's been encountered. Entries here are not yet committed to long-term knowledge. In the brain, this maps to hippocampal engrams — fast, sparse, specific to episodes.

**Two-stage promotion gate (gate.py)** — The decision mechanism that takes candidate experiences from the buffer and either commits them to the slow layer or discards/shelves them. It has two stages, mirroring the brain's molecular "timer cascade":

- **Stage 1 (cheap, runs constantly):** Acts like the early checkpoints in the brain's memory cascade — thalamic/hippocampal gating that quickly flags experiences as worth keeping or not. Two independent signals can trigger it: frequency ("this keeps happening, it's probably a real pattern") or persistent surprise ("I keep getting this wrong even though I've seen it many times — something fundamental is missing"). This is a cheap filter — just counting and trend-fitting — so it can run on everything in the buffer without significant cost.

- **Stage 2 (expensive, only for candidates):** Acts like the late-stage consolidation process in the brain — cortical protein synthesis, synaptic remodeling, engram maturation. It tentatively integrates the candidate knowledge and runs two checks: (a) does the new knowledge actually reduce prediction error for this type of experience (is it real learning, not noise?), and (b) does it damage accuracy on already-consolidated knowledge (am I forgetting something important?). If both pass, the knowledge is committed and Fisher importance masks are computed to protect it from future interference. This is computationally expensive, so it only fires on candidates that survived Stage 1.

**Fisher importance masks (gate.py)** — Each time a cluster of knowledge is committed, the system computes which weights were most important for that cluster (by looking at the squared gradients during the integration fine-tune). This produces a per-weight importance score — the Fisher information diagonal. In future training, an EWC-style penalty pushes the model to avoid changing high-importance weights unless it's really necessary. Multiple clusters' masks are merged by taking the max importance per weight across all clusters — so a weight that's important for any cluster is protected globally.

### The Feedback Loops

1. **Explore loop:** The model trains on incoming data → prediction error is computed → experience is logged to the buffer → fast layer (not in v1) updates immediately.
2. **Consolidation loop:** Periodically, the gate runs over the buffer → Stage 1 flags candidates → Stage 2 validates and commits → Fisher masks are stored → slow layer gains new protected knowledge.
3. **Recall-destabilization (deferred to v3):** When consolidated knowledge is contradicted by new evidence, its protection is temporarily lowered so it can be updated. Not built yet.

---

## 2. The Analogy Table

| Brain Component | AI Analogue | v1 Status |
|----------------|-------------|-----------|
| Hippocampus (fast encoding of new episodes) | Episodic buffer + future fast-weight layer | Buffer: BUILT. Fast weights: DEFERRED |
| Cortex (slow, dense, consolidated knowledge) | Slow layer (the CNN) | BUILT |
| Hippocampal engrams (specific episodic traces) | BufferEntry objects | BUILT |
| Cortical engrams (abstracted, generalized patterns) | Fisher-protected weight configurations | BUILT |
| Molecular timer cascade (staged promote/demote) | Two-stage promotion gate | BUILT |
| Sleep replay / consolidation | Periodic gate evaluation | BUILT |
| Synaptic consolidation (weight-level protection) | EWC-style Fisher importance penalty | BUILT |
| Systems reconsolidation (recall destabilizes memory) | Fisher-mask selective unlocking | DEFERRED |
| Neurogenesis / resource cost for reconsolidation | Destabilization budget/cooldown | DEFERRED |
| Predictive coding / surprise signals | Prediction error history per cluster | BUILT |

---

## 3. What Makes This Different From Standard Approaches

**Vs. plain EWC:** EWC computes one importance mask per task (or one cumulative mask over everything) and applies it uniformly. The two-stage gate adds a *candidacy filter* before anything gets committed — not every experience gets to influence the protected weights, only those that prove they're worth the cost. It also decouples *what gets considered* (Stage 1) from *what gets committed* (Stage 2), whereas EWC commits everything from the current task unconditionally.

**Vs. experience replay:** Pure replay just retrains on old data periodically to avoid forgetting. This architecture adds a structured *importance-weighting* (Fisher masks) and a *destabilization mechanic* (deferred) that replay alone doesn't model.

**Vs. Progressive Neural Networks:** PNN grows a new column per task — size grows without bound. This architecture keeps a single network and protects weights within it.

**Vs. foundation-model approaches (Voyager, AdA):** Those use frozen brains + external memory (Voyager: code library; AdA: in-context attention window). This architecture updates weights themselves online, which is harder but more flexible — the agent's actual understanding can grow, not just its notes.

---

## 4. Scope Boundaries for v1

**In scope:**
- Small CNN as the slow layer
- Episodic buffer with class-based clustering
- Two-stage promotion gate (frequency OR persistent-surprise → Fisher-validated integration)
- EWC-style Fisher importance penalty on future training
- Three-way comparison: naive vs. EWC vs. two-stage gate
- ACC and BWT metrics

**Explicitly deferred:**
- Fast-weight layer / fast-slow split efficiency question
- Recall-destabilization mechanics
- Curiosity reward / intrinsic motivation
- World models / mental simulation
- Foundation model / common sense
- 3D exploration environment
- RL loop
- Self-attention or transformer layers

---

## 5. Environment Setup

**OS:** Linux (Ubuntu 24.04.4 recommended)
**GPU:** AMD RX 7900 GRE (gfx1100 architecture, officially supported by ROCm 7.2.4+)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm<version>
```

Check pytorch.org for the exact current ROCm wheel tag — it moves with ROCm releases. Verify with:

```python
import torch
print(torch.cuda.is_available())          # should be True
print(torch.cuda.get_device_name(0))      # should show your AMD GPU
```

ROCm builds of PyTorch expose through the `torch.cuda` namespace. For v1 (MNIST-scale), CPU is sufficient — don't over-invest in GPU setup.

---

## 6. Project Layout

```
project/
  data.py       # Split-MNIST task stream
  model.py      # slow network (small CNN)
  buffer.py     # episodic buffer + clustering
  gate.py       # stage 1 + stage 2 promotion logic
  baselines.py  # naive fine-tune, standard EWC
  train.py      # experiment harness
  metrics.py    # ACC / forgetting computation
```

---

## 7. data.py — Split-MNIST Task Stream

**5 sequential tasks**, each introducing 2 digit classes:

| Task | Classes |
|------|---------|
| 0    | 0, 1    |
| 1    | 2, 3    |
| 2    | 4, 5    |
| 3    | 6, 7    |
| 4    | 8, 9    |

Export a function that returns an iterable of tasks, where each task provides train and test torch DataLoaders. Standard MNIST pre-processing (28x28 grayscale, normalize to [0,1] or standardize).

Standard continual-learning benchmark with known baseline numbers to sanity-check against.

---

## 8. model.py — Slow Network

Small CNN (no fast-weight adapter):

```
Input:  (1, 28, 28)
  → Conv2d(1, 32, kernel=3, padding=1) + ReLU
  → Conv2d(32, 64, kernel=3, padding=1) + ReLU
  → MaxPool2d(2)
  → Flatten
  → Linear(14*14*64, 256) + ReLU
  → Linear(256, 10)
```

Export: `create_model() -> nn.Module`. This is the "slow" layer for v1.

---

## 9. buffer.py — Episodic Buffer

```python
class BufferEntry:
    cluster_id: int              # v1: cluster = task/class id (0-9 for MNIST digits)
    inputs: List[Tensor]         # accumulated inputs
    labels: List[Tensor]         # corresponding labels
    pred_error_history: List[float]  # prediction errors recomputed each recheck
    seen_count: int              # number of times this cluster has been observed

class EpisodicBuffer:
    def __init__(self, max_size: int):
        self.entries: Dict[int, BufferEntry] = {}

    def add(self, cluster_id: int, inputs: Tensor, labels: Tensor):
        """Add or append to an existing cluster entry."""

    def recompute_errors(self, model: nn.Module):
        """Run model on each cluster's data, update pred_error_history."""

    def get_candidates(self) -> List[BufferEntry]:
        """Return all entries for stage-1 evaluation."""
```

**Clustering simplification:** v1 clusters by ground-truth class/digit label (0-9). This sidesteps the "what counts as one belief" open question so you can test the gate mechanism first. Cluster IDs map directly to digit classes.

**Buffer forgetting:** The buffer itself needs a max_size and eviction policy. Most experiences should eventually be evicted, not promoted. Use a simple FIFO or recency-based eviction per cluster.

---

## 10. gate.py — Two-Stage Promotion

### Stage 1 — Cheap Filter

Runs periodically (e.g. every N batches) over all buffer entries.

```python
def is_candidate(entry: BufferEntry, freq_threshold: int, persist_window: int) -> bool:
    """
    Returns True if entry qualifies for stage-2 evaluation.

    frequent = entry.seen_count >= freq_threshold
    persistently_surprising = errors are NOT decreasing over the last `persist_window` observations

    return frequent or persistently_surprising
    """
```

`is_decreasing` — fit a simple linear trend (e.g. scipy.stats.linregress or manual least-squares) on `pred_error_history[-persist_window:]`. If the slope is non-negative (flat or rising), the model isn't explaining this away on its own → persistently surprising.

**`frequent or persistently_surprising`** (OR, not AND): "this keeps happening" and "this remains unexplained" are two independent reasons to consolidate. Using AND would miss real cases of each.

**Configurable parameters:** `freq_threshold` (e.g. 50), `persist_window` (e.g. 10).

---

### Stage 2 — Expensive Validation

```python
def validate_and_commit(
    model: nn.Module,
    candidate: BufferEntry,
    fisher_masks: Dict[int, Tensor],  # cluster_id -> Fisher diagonal
    eps_gain: float,
    eps_forget: float,
    replay_sample: DataLoader,
    lr: float,
    steps: int,
    lambda_ewc: float,
) -> bool:
    """
    1. Create a shadow copy of model
    2. Fine-tune shadow on candidate's data (steps=K, lr=low_lr)
    3. Compute gain = old_error(model, candidate) - new_error(shadow, candidate)
    4. Compute forgetting = eval_replay_sample(model) - eval_replay_sample(shadow)
    5. If gain > eps_gain AND forgetting < eps_forget:
       - Copy shadow weights into model
       - Compute Fisher diagonal for this cluster (harvest gradients from the fine-tune pass)
       - Store in fisher_masks[candidate.cluster_id]
       - Return True
    6. Otherwise return False (candidate stays in buffer, can retry later)
    """
```

**Key details:**
- `compute_fisher_diag` reuses gradients already computed during the fine-tune pass — do NOT recompute from scratch. After the last fine-tune step on the shadow model, compute the Fisher diagonal as the mean of squared gradients w.r.t. each parameter, averaged over the candidate's data.
- `eval_replay_sample` runs the model on a small fixed sample drawn from all previously-committed clusters to check for forgetting.
- `eps_gain` and `eps_forget` are tunable thresholds (e.g. 0.01 and 0.05).

### EWC-style penalty applied during future training

Every training step (for every config — EWC and two-stage gate) adds to the loss:

```python
def ewc_penalty(model, fisher_masks, theta_star):
    """
    loss += lambda_ewc * sum_over_clusters(
        sum_over_params( fisher_masks[cluster][i] * (theta[i] - theta_star[cluster][i])**2 )
    )
    """
    # Merge masks: for each parameter, take the running MAX Fisher importance across all clusters
    # (standard multi-task EWC accumulation)
    # theta_star[cluster] is the parameter state when that cluster was committed
```

The merged mask means: `F_merged[i] = max(F_mask_0[i], F_mask_1[i], ...)` — the highest importance any cluster assigned to parameter i.

---

## 11. baselines.py — Comparison Configs

### Naive fine-tuning (forgetting floor)
```python
def train_naive(model, tasks, epochs_per_task, lr):
    """Plain sequential SGD fine-tuning with no protection at all."""
    for task_id, (train_loader, test_loader) in enumerate(tasks):
        # Standard supervised training on this task's data
        pass
```

### Standard EWC (the bar to beat)
```python
def train_ewc(model, tasks, epochs_per_task, lr, lambda_ewc):
    """
    One cumulative Fisher penalty over all tasks so far.
    No gate, no candidacy filter — the penalty applies unconditionally.
    Fisher mask is computed AFTER each task using the full task's data.
    """
    # After each task, compute Fisher diagonal on that task's data
    # Accumulate via running max across tasks
    # Apply EWC penalty during all future training
```

---

## 12. train.py — Experiment Harness

```python
def run_experiment(config_name, seed, ...):
    """Run one full experiment for a given config and seed."""
    set_seed(seed)
    model = create_model().to(device)
    tasks = get_split_mnist_tasks(batch_size=128)

    if config_name == "naive":
        train_naive(model, tasks, ...)
    elif config_name == "ewc":
        train_ewc(model, tasks, ...)
    elif config_name == "two_stage_gate":
        train_two_stage_gate(model, tasks, ...)

    # Log ACC and BWT after each task
    return metrics_history

if __name__ == "__main__":
    configs = ["naive", "ewc", "two_stage_gate"]
    seeds = [42, 43, 44, 45, 46]
    for config in configs:
        for seed in seeds:
            run_experiment(config, seed)
```

### Training loop for two_stage_gate

```python
def train_two_stage_gate(model, tasks, epochs_per_task, ...):
    buffer = EpisodicBuffer(max_size=10000)
    fisher_masks = {}
    theta_star = {}  # cluster_id -> parameter state at commit time

    for task_id, (train_loader, test_loader) in enumerate(tasks):
        for epoch in range(epochs_per_task):
            for batch_x, batch_y in train_loader:
                # Standard forward/backward with EWC penalty applied
                loss = ce_loss(model(batch_x), batch_y)
                loss += ewc_penalty(model, fisher_masks, theta_star)
                loss.backward()
                optimizer.step()

                # Log this experience to buffer (use batch's label as cluster_id)
                # This is simplified — real per-sample logging would be finer-grained
                buffer.add(cluster_id=batch_y, inputs=batch_x, labels=batch_y)

            # Run promotion gate every N batches / every epoch
            buffer.recompute_errors(model)
            for entry in buffer.get_candidates():
                if is_candidate(entry, freq_threshold=50, persist_window=10):
                    # Build a small replay sample from committed clusters
                    replay_sample = build_replay_sample(buffer, fisher_masks)
                    validated = validate_and_commit(
                        model, entry, fisher_masks,
                        eps_gain=0.01, eps_forget=0.05,
                        replay_sample=replay_sample,
                        lr=1e-4, steps=5, lambda_ewc=lambda_ewc
                    )
                    if validated:
                        # Candidate was committed, remove from buffer or mark as such
                        pass

        # After each task: record metrics (ACC, BWT)
        record_metrics(model, tasks, task_id)
```

### Config values to start with (tunable):

| Parameter | Initial value |
|-----------|--------------|
| epochs_per_task | 2 |
| learning_rate | 1e-3 (SGD with momentum 0.9) |
| batch_size | 128 |
| lambda_ewc | 0.1 (try 0.01, 0.1, 1.0) |
| freq_threshold | 50 |
| persist_window | 10 |
| eps_gain | 0.01 |
| eps_forget | 0.05 |
| stage2_fine_tune_steps | 5 |
| stage2_fine_tune_lr | 1e-4 |
| promo_gate_interval | every epoch (i.e. every ~469 batches for 60k/128) |
| buffer_max_size | 10000 |
| replay_sample_size | 512 |

---

## 13. metrics.py — Evaluation

### ACC (Average Accuracy)

After each task `t`, compute accuracy on all task test sets seen so far (tasks 0..t). Then average them:

```python
def compute_acc(model, tasks, up_to_task):
    accuracies = []
    for task_id in range(up_to_task + 1):
        acc = evaluate(model, tasks[task_id].test_loader)
        accuracies.append(acc)
    return mean(accuracies)
```

### BWT (Backward Transfer / Forgetting)

For each task `i`, compare accuracy right after it was first learned vs. accuracy now (after task `t`):

```python
def compute_bwt(model, tasks, task_accs_after_learning):
    """
    task_accs_after_learning[i] = accuracy on task i right after task i finished training
    task_accs_now[i] = accuracy on task i at the current time

    BWT = mean over i of (task_accs_now[i] - task_accs_after_learning[i])
    Negative BWT = forgetting. Zero = no forgetting. Positive = backward transfer (improvement).
    """
```

### Logging

After each task, log: `task_id, config_name, seed, ACC, BWT, and per-task accuracies`.

---

## 14. Run Configuration

- **Same architecture** across all 3 configs
- **Same task order** (0/1 → 2/3 → 4/5 → 6/7 → 8/9)
- **5 seeds** per config (42, 43, 44, 45, 46)
- **3 configs:** naive, ewc, two_stage_gate
- Output: single CSV or JSON file with all metric snapshots

**Generating the output plot:** ACC curves (x=task_id, y=ACC, 3 lines with std-dev bands) and BWT curves (x=task_id, y=BWT, same configs). matplotlib is fine.

---

## 15. Expected Outcomes & Interpretation

| Outcome | Interpretation |
|---------|---------------|
| Gate beats EWC on BWT, ACC comparable or better | Staged gating has real value. Proceed to v2 (add fast-weight layer). |
| Gate ≈ EWC | Added complexity not earning its cost at this scale. Could be task boundaries too clean — test with noisier stream. Or machinery genuinely unnecessary. |
| Gate worse than EWC | Likely tuning issue: thresholds too loose (behaves like naive) or too strict (nothing promotes), or replay sample unrepresentative. |
| High variance across seeds | Mechanism unstable; need more seeds or slightly larger benchmark. |

---

## 16. Implementation Order

1. **model.py** — Create the CNN, verify it trains on a single task
2. **data.py** — Split-MNIST task stream, verify loader shapes
3. **metrics.py** — ACC/BWT functions, verify on dummy data
4. **baselines.py** — naive first (easiest), then EWC
5. **buffer.py** — with cluster logic
6. **gate.py** — stage 1 + stage 2
7. **train.py** — wire everything together

---

## 17. Design Constraints Carried Forward

- **No self-attention or transformer layers.** Small CNN only.
- **No foundation model.** No frozen pretrained weights anywhere.
- **Fisher info harvested as byproduct** of the stage-2 fine-tune pass, not computed separately.
- **EWC-style penalty** uses running-max Fisher accumulation across clusters.
- **Clustering by class label** is a simplification — acceptable for v1 since it unblocks the gate test.

---

## 18. Quick Reference: Function Signatures

```python
# data.py
def get_split_mnist_tasks(batch_size: int = 128) -> List[Tuple[DataLoader, DataLoader]]:
    """Returns list of (train_loader, test_loader) per task."""

# model.py
def create_model() -> nn.Module:
    """Returns small CNN: Conv->Conv->Pool->FC->FC, 10 output classes."""

# buffer.py
class BufferEntry:
    cluster_id: int
    inputs: List[Tensor]
    labels: List[Tensor]
    pred_error_history: List[float]
    seen_count: int

class EpisodicBuffer:
    def __init__(self, max_size: int): ...
    def add(self, cluster_id: int, inputs: Tensor, labels: Tensor): ...
    def recompute_errors(self, model: nn.Module): ...

# gate.py
def is_candidate(entry: BufferEntry, freq_threshold: int, persist_window: int) -> bool: ...
def validate_and_commit(model, candidate, fisher_masks, eps_gain, eps_forget, replay_sample,
                        lr, steps, lambda_ewc) -> bool: ...
def compute_fisher_diag(model, data_loader) -> Dict[str, Tensor]:
    """Returns per-parameter squared gradients averaged over data."""
def ewc_penalty(model, fisher_masks: Dict[int, Dict[str, Tensor]],
                theta_star: Dict[int, Dict[str, Tensor]], lambda_ewc: float) -> Tensor: ...

# baselines.py
def train_naive(model, tasks, epochs_per_task, lr): ...
def train_ewc(model, tasks, epochs_per_task, lr, lambda_ewc): ...

# metrics.py
def compute_acc(model, tasks, up_to_task: int) -> float: ...
def compute_bwt(model, tasks, task_accs_after_learning: List[float]) -> float: ...

# train.py
def run_experiment(config_name: str, seed: int, ...) -> Dict: ...
```
