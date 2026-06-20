# Handoff for Next Stage

## What Was Done

1. **Full v1 prototype built.** All 7 modules: model, data, metrics, baselines, buffer, gate, train. Each independently verifiable.

2. **Environment set up.** Python 3.12 + uv + PyTorch 2.9.1 for ROCm 7.2.4 on AMD RX 7900 GRE.

3. **Two diagnostics completed:**
   - Fisher ranking vs empirical importance (diagnose_fisher.py)
   - Per-layer stratified freezing (diagnose_freeze.py)

4. **Key finding:** Diagonal Fisher is directionally correct (ρ=0.43 overall, ρ=0.96 for conv1) but structurally insufficient for per-parameter protection at 3.2M params. Distributed-representation collapse occurs regardless of normalization or λ. EWC ≈ naive at this scale.

5. **Bug fixed:** Optimizer recreated per-batch in train_two_stage_gate (momentum reset). Moved to per-task creation.

## What Remains

1. **The two-stage gate has NOT been experimentally validated.** The `validate_and_commit` function, `is_candidate` filter, and `train_two_stage_gate` pipeline are all wired and ready. But the full 3-config × 5-seed experiment has not been run.

2. **Threshold tuning** for the gate (eps_gain, eps_forget, freq_threshold, persist_window) is untested. Current values are spec defaults.

3. **The replay-sample check** (the gate's actual forgetting-prevention mechanism) needs validation that it can prevent distributed collapse where the Fisher mask alone cannot.

## Key Risks for Next Agent

1. **Gate may also fail.** If the replay-sample check cannot prevent forgetting (because the 5-step fine-tune on candidate data is too brief, or the forgetting threshold is too loose), even the gate ≈ naive. This cannot be predicted — needs experimental run.

2. **The EWC baseline is dead.** Don't invest more time trying to make EWC work at this scale. It's structurally limited by the diagonal approximation in overparameterized nets. If you need a regularization-based baseline, use SI (Synaptic Intelligence) or MAS (Memory-Aware Synapses).

3. **The comparison framing matters.** The hypothesis says "gate beats EWC." Since EWC ≈ naive, the actual comparison is "gate beats naive." Document this explicitly in any write-up.

## Files to Examine Next

| File | Purpose |
|------|---------|
| `train.py` | Entry point. `run_experiment` dispatches configs. `train_two_stage_gate` is the main gate pipeline. |
| `gate.py` | `validate_and_commit` — this is the core mechanism. Replay sample check is the forgetting-prevention mechanism. |
| `baselines.py` | `train_ewc` — uses per-layer normalized Fisher. Verified to compute correctly but produces naive-identical results. |
| `diagnose_fisher.py` | Fisher ranking experiment. Run this first to reproduce the correlation finding. |
| `diagnose_freeze.py` | Per-layer freeze experiment. Run this to reproduce the distributed-collapse finding. |

## Quick Start

```bash
source .venv/bin/activate
python3 train.py --help        # see all options
python3 train.py --configs naive ewc two_stage_gate --seeds 42 43 44 45 46
```

## Dependencies

Current `pyproject.toml` lists torch/torchvision/scipy/matplotlib/numpy as dependencies. ROCm torch is installed from local WHL files (not from PyPI). If recreating the environment, download from:

```
https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/
```

For CPU-only (which suffices for MNIST), install standard torch from PyPI.
