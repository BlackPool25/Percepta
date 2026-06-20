# Architecture Implemented (v1 Prototype)

**Per spec:** `PROTOTYPE_BUILD.md` — all 7 modules built, each independently verifiable.

## Project Layout

```
percepta/
  model.py      # SlowCNN: 3.2M params, 10 output classes
  data.py       # Split-MNIST: 5 tasks, 2 classes each
  metrics.py    # ACC, BWT, evaluate
  baselines.py  # train_naive, train_ewc, compute_fisher_diag, ewc_penalty
  buffer.py     # EpisodicBuffer with class-based clustering, FIFO eviction
  gate.py       # is_candidate, validate_and_commit, theta_star_merged
  train.py      # run_experiment, build_replay_sample, plotting
  diagnose_fisher.py   # Fisher vs empirical importance diagnostic
  diagnose_freeze.py   # Per-layer stratified freeze experiment
```

## Module Details

### model.py
- `SlowCNN`: Conv2d(1→32) → ReLU → Conv2d(32→64) → ReLU → MaxPool2d(2) → Flatten → Linear(12544→256) → ReLU → Linear(256→10)
- 3,232,906 parameters
- `create_model()` → `nn.Module`

### data.py
- `get_split_mnist_tasks(batch_size=128)` returns `list[(train_loader, test_loader)]`
- Tasks: `[(0,1), (2,3), (4,5), (6,7), (8,9)]`
- Standard MNIST normalization: mean=0.1307, std=0.3081

### metrics.py
- `evaluate(model, loader, device)` → accuracy
- `compute_acc(model, tasks, up_to_task, device)` → mean accuracy across tasks 0..up_to_task
- `compute_bwt(task_accs_now, task_accs_after)` → mean forgetting

### baselines.py
- `train_naive` — plain SGD fine-tuning, no protection
- `train_ewc` — standard EWC with per-layer normalized Fisher masks
- `compute_fisher_diag` — mean of squared gradients over data
- `normalize_fisher_mask` — per-layer max normalization to [0,1]
- `ewc_penalty` — λ · Σ F_i · (θ_i − θ*_i)²

### buffer.py
- `BufferEntry`: cluster_id, inputs, labels, pred_error_history, seen_count
- `EpisodicBuffer`: per-class entries, FIFO eviction at max_size
- `recompute_errors`: runs model on each cluster, updates pred_error_history

### gate.py
- `is_candidate`: frequency threshold OR persistent-surprise filter (linear trend on prediction errors)
- `validate_and_commit`: shadow model fine-tune → gain/forgetting check → conditional commit
- Fisher diagonal harvested as byproduct of fine-tune pass

### train.py
- `run_experiment`: dispatches to naive/ewc/two_stage_gate, returns metric history
- `build_replay_sample`: balanced sample from committed clusters
- `train_two_stage_gate`: full pipeline with buffer logging, promotion gate, metrics recording
- Output: CSV + JSON results, ACC and BWT plots
- Default: 3 configs × 5 seeds (42-46)

## Known Bug

`train_two_stage_gate` in `train.py` had the optimizer recreated on every batch (resetting momentum). **Fixed** — now created once per task.
