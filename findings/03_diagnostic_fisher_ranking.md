# Diagnostic 1: Fisher Ranking vs Empirical Importance

**File:** `diagnose_fisher.py`

## Question

Is the diagonal Fisher information directionally correct at this scale (3.2M params, CNN on Split-MNIST)? Does a higher Fisher value actually indicate a more important parameter?

## Method

1. Train model on task 0 (digits 0/1) to convergence (99.8% test acc)
2. Compute Fisher diagonal on task 0 training data
3. Group parameters into 362 functional units (conv filters + FC neurons)
4. For each group: zero it out, measure accuracy drop on task 0 test set
5. Compute Spearman ρ between (mean Fisher per group) × (accuracy drop)

## Results

| Metric | Value |
|--------|-------|
| Overall Spearman ρ | 0.434 (p < 0.001) |
| Pearson r(log(Fisher), importance) | 0.965 |
| Total groups | 362 |

### Per-layer breakdown

| Layer | Groups | ρ | Notes |
|-------|--------|---|-------|
| conv1 | 32 filters | **0.962** | Very strong — early filters have clean role separation |
| conv2 | 64 filters | **0.693** | Strong — some feature entanglement |
| fc1 | 256 neurons | **0.341** | Moderate — dense layer, heavy entanglement |
| fc2 | 10 output neurons | N/A (only 2 non-zero) | Too few groups for correlation |

## Interpretation

**Fisher IS directionally correct.** The ranking is valid — higher Fisher genuinely means higher importance. But the signal strength varies dramatically by layer:
- Early conv layers: excellent (ρ=0.96) — filters have localized, independent roles
- Late dense layers: weak (ρ=0.34) — parameters are heavily entangled; individual importance is hard to isolate

This means: the Fisher ranking is trustworthy in aggregate, but its reliability is layer-dependent. Any mechanism relying on Fisher-based parameter selection must account for this layer imbalance.
