# EWC Baseline Analysis

## The Problem

The EWC baseline produces results identical to naive fine-tuning across λ ∈ [0.1, 1000], even with per-layer normalized Fisher masks.

### Root Cause

Three interacting factors:

1. **Fisher at convergence is near-zero.** After training on a task to 99.8% accuracy, gradients are tiny (mean Fisher ≈ 3.95e-8). The squared gradient signal is structurally small.

2. **Per-layer normalization fixes the scale but not the distributed-representation problem.** Even with normalized Fisher masks where each layer's max = 1.0, the EWC penalty is insufficient because the penalty acts per-parameter, but task performance depends on collective parameter interactions.

3. **Freezing experiments confirmed distributed collapse.** Even freezing 25% of each layer's highest-Fisher parameters (including fc2's output neurons) and enforcing zero change on them, task 0 accuracy still drops to 0%. The representation is distributed — protecting individual high-Fisher params can't preserve it.

### Historical Context

This is a known limitation of diagonal Fisher EWC in overparameterized networks. The original EWC paper (Kirkpatrick et al. 2017) demonstrated results on smaller networks (typically 2-3 layer MLPs or moderate CNNs). The diagonal Fisher approximation degrades as network scale increases because:

- Individual parameter importance is swamped by interactions
- The diagonal ignores off-diagonal Fisher entries (which capture parameter correlations)
- Distributed representations mean no single parameter carries enough information to be individually critical

## What This Means for v1

The comparison is not "gate vs EWC vs naive" as the hypothesis states. It is effectively **"gate vs naive"** because EWC ≈ naive at this scale.

This is still informative:
- If the gate beats naive, it demonstrates that the **replay-sample check** (the gate's actual forgetting-prevention mechanism) provides real protection where per-parameter regularization cannot
- If the gate ≈ naive, the staged mechanism adds no value

The hypothesis (Section 1 of PROTOTYPE_BUILD.md) is technically underdetermined by this experiment design because the EWC baseline is structurally incapable of outperforming naive at this scale. To properly test "gate beats EWC," one would need either:
- A smaller model where diagonal Fisher is denser
- A different baseline (SI, MAS, or replay-based)
- A different metric (forgetting rate on individual examples rather than per-task accuracy)
