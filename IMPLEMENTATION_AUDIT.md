# Percepta v1 — Implementation Audit

**Status:** All modules built. Debugging EWC baseline and gate validation.

---

## What Exists (All Working Independently)

| Module | Status | Notes |
|--------|--------|-------|
| `model.py` | ✅ verified | SlowCNN: 3.2M params, Conv1→Conv2→Pool→FC→FC, 10 outputs |
| `data.py` | ✅ verified | 5 Split-MNIST tasks, correct label filtering |
| `metrics.py` | ✅ verified | `evaluate`, `compute_acc`, `compute_bwt` |
| `buffer.py` | ✅ verified | FIFO eviction, per-class clustering, error recomputation |
| `gate.py` | ✅ built | Both stages implemented, not end-to-end tested |
| `baselines.py` | ⚠️ needs review | EWC baseline has a fundamental issue |
| `train.py` | ⚠️ needs review | Optimizer recreated per-batch (wasteful), gate wiring |

---

## The EWC Problem

### What I Found

The EWC baseline produces identical results to naive fine-tuning across `lambda_ewc = [1, 10, 50, 100, 1000]`. The reason:

**The diagonal Fisher is near-zero for this setup.**

Measured values:
- Fisher after convergence: **mean ≈ 1.6e-8** per parameter
- Fisher during early training: **mean ≈ 0.012** per parameter

Even with the "during training" Fisher, the penalty doesn't prevent forgetting at reasonable lambda values. Only at `lambda=100` did I see any effect (partial retention of task 0, at the cost of not learning task 4).

### Why This Happens

1. **Overparameterization.** The CNN has 3.2M parameters for a 2-class MNIST subtask. Most parameters are in flat regions of the loss — changing them individually doesn't affect the loss much. The diagonal Fisher only measures per-parameter curvature, missing interactions.

2. **Fisher measures local curvature.** After convergence, gradients are near-zero. The Fisher (E[squared gradients]) is the expected squared gradient under the model's predictive distribution. For a well-fitting model with 3.2M parameters on 12k training examples, many weights have near-zero squared gradients.

3. **The "during training" fix is conceptually wrong.** Computing Fisher on task k's data during task k training measures the gradient of the *current loss*, not the *importance for retention*. These gradients reflect what the model is currently trying to change, not what it should protect.

4. **EWC's diagonal approximation is weakest for overparameterized nets.** This is a known limitation in the literature — EWC works best when parameters have clearly differentiated roles.

### What We Could Try Instead

These are options, not recommendations — you decide:

- **Shrink the model.** A 2-layer CNN with 32→64 filters → the smaller the model, the denser the importance signal. E.g., Conv1(1→8) + Conv2(8→16) + FC(256→10) would have ~100K params instead of 3.2M.
- **Use Synaptic Intelligence (SI)** instead of EWC. SI accumulates importance during the *whole trajectory* of learning a task, not just at the endpoint. It captures which parameters moved a lot and contributed to loss reduction. This is usually more robust than EWC for overparameterized nets.
- **Use memory replay** as the forgetting-prevention mechanism instead of regularization. With a small buffer of old examples, the model trains jointly on new data + replayed old data. This is simpler and often more effective than EWC for small-scale CL.
- **Accept that EWC doesn't buy much here** and treat it as a baseline to beat. If the two-stage gate beats naive even by a small margin, that's evidence the gate works. The "EWC ≈ naive" outcome is itself a finding.

---

## The Optimizer Bug in train.py

In `train_two_stage_gate`, the optimizer is recreated on every batch:

```python
for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
```

This resets momentum on every batch. Should be created once per task or per epoch.

---

## The Gate Validation Logic

`validate_and_commit` creates a shadow copy of the model, fine-tunes on candidate data, checks gain and forgetting. This is correct conceptually.

Open questions:
- **Fisher for the gate.** `compute_fisher_diag` runs after the shadow has been fine-tuned on candidate data. If the model is near-converged on the candidate, the Fisher will be near-zero, same as the EWC problem. The gate should probably use the training-phase Fisher (accumulated during the fine-tune).
- **Replay sample.** Currently samples equally from all committed clusters. Need to verify this is representative.
- **Thresholds.** `eps_gain=0.01, eps_forget=0.05` are guesses. No tuning done.

---

## Next Decision to Make

The fork is:

1. **Keep the current architecture** (3.2M CNN, diagonal Fisher EWC baseline, gate with Fisher-based validation). Accept that EWC ≈ naive and the two-stage gate comparison becomes "gate vs. naive." If gate beats naive, that's sufficient for v1.

2. **Switch to a smaller model** so the Fisher signal is denser and EWC has a fighting chance. This is more principled but requires re-verifying the split-MNIST baseline numbers.

3. **Replace EWC with a replay-based baseline.** Give both baselines and the gate a small episodic memory buffer. The comparison becomes "naive + replay vs. EWC + replay vs. two-stage gate." This is more standard in modern continual learning.

What do you want to do?
