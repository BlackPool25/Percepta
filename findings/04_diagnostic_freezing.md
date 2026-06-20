# Diagnostic 2: Per-Layer Stratified Freezing

**File:** `diagnose_freeze.py`

## Question

If we protect the top-K% highest-Fisher parameters within EACH layer (guaranteeing fc2's most important output neurons are locked), does task 0 accuracy survive training on task 1?

This distinguishes between:
- **Selection bias** (global top-1% misses fc2 entirely because conv1/conv2 have wider Fisher range)
- **Distributed collapse** (even locking fc2's most important neurons can't preserve task-0 accuracy because the representation is distributed across millions of interacting params)

## Method

1. Train on task 0, compute per-layer normalized Fisher masks
2. For each layer independently, select top-K% of parameters by Fisher magnitude
3. Freeze selected parameters (restore to task-0 values after each task-1 gradient step)
4. Train on task 1 for 2 epochs
5. Measure task 0 and task 1 test accuracy

## Results

| Freeze % | Params frozen | t0 acc | t1 acc |
|----------|---------------|--------|--------|
| 0.1% | 3,232 | 0.0000 | 0.9755 |
| 0.5% | 16,162 | 0.0000 | 0.9760 |
| 1.0% | 32,327 | 0.0000 | 0.9726 |
| 2.0% | 64,656 | 0.0000 | 0.9726 |
| 5.0% | 161,644 | 0.0000 | 0.9736 |
| 10.0% | 323,290 | 0.0000 | 0.9740 |
| 25.0% | 808,226 | 0.0000 | 0.9736 |

**Note:** fc2 (output layer, 2570 values) had its top-K% frozen at every level. The highest-Fisher output neurons were locked to task-0 values throughout.

## Conclusion

**Distributed collapse, confirmed.** Even with per-layer stratified freezing including fc2's most important output neurons, task 0 accuracy drops to 0%. The representation of digit classes is distributed across millions of interacting parameters — no per-parameter selection can isolate it.

### Per-layer change analysis (at 25% freeze)

| Layer | Frozen MSE | Free MSE | Interpretation |
|-------|-----------|----------|----------------|
| conv1 | 0.00000000 | 0.000759 | Heavy repurposing of early filters |
| conv2 | 0.00000000 | 0.000029 | Moderate feature change |
| fc1 | 0.00000000 | 0.000001 | Dense layer barely changes (but still forgets) |
| fc2 | 0.00000000 | 0.000902 | Output weights shift to fit task 1 |

The frozen params never change (MSE=0 by construction) but the **free** params in every layer — even fc1 where free_mse is only 0.000001 — collectively disrupt the task-0 representation enough to cause complete forgetting.

### Architectural implications

1. **Diagonal Fisher is insufficient** for per-parameter protection at this scale. The distributed nature of neural network representations means that small changes across millions of low-Fisher parameters can collapse a learned task. This is a structural limitation of the diagonal approximation, not a tuning issue.

2. **EWC ≈ naive** for this setup. The EWC baseline cannot outperform naive fine-tuning because the diagonal Fisher penalty cannot prevent distributed collapse regardless of λ or normalization scheme.

3. **The gate's Fisher mask has the same limitation.** The mask from `compute_fisher_diag` in Stage 2 is directionally correct but cannot selectively protect committed knowledge. The gate's actual forgetting-prevention mechanism is the **replay-sample check** (comparing old vs new loss on committed clusters), not the Fisher mask.

4. **D5/D6 survive with caveats.** Fisher-based addressing for recall-destabilization (v3) will be approximate, not precise. Belief bleed is inherent — consistent with the biology caveat already documented ("precision loss in nearby memories is expected and has biological precedent").

## What This Does NOT Mean

- Fisher is not noise — the ranking IS valid (diagnostic 1, ρ=0.434 overall, ρ=0.962 for conv1)
- The gate mechanism is not invalidated — the replay-sample check provides the forgetting detection, which is a fundamentally different mechanism from per-parameter protection
- Per-layer normalization is not useless — it just can't fix the distributed-representation problem
