# Handoff for Next Stage (Updated: v1.1 Complete)

## What Was Done (v1)

1. **Full v1 prototype built.** All 7 modules: model, data, metrics, baselines, buffer, gate, train.

2. **Environment set up.** Python 3.12 + uv + PyTorch 2.9.1 for ROCm 7.2.4 on AMD RX 7900 GRE.

3. **Two diagnostics completed:** Fisher ranking vs empirical importance, per-layer stratified freezing.

4. **Key finding:** Diagonal Fisher is directionally correct but structurally insufficient. EWC ≈ naive at this scale.

## What Was Done (v1.1 — 6/20/2026)

5. **Gate validated experimentally across 5 seeds.** 3-config × 5-seed benchmark complete.

6. **Architecture improvements:**
   - Interleaved replay during Stage 2 consolidation (joint candidate+replay shadow fine-tune)
   - Ongoing replay in main training loop (every batch trains on new data + replayed committed data)
   - Core-set protected buffer (100 examples per committed cluster survive FIFO eviction)
   - Sparse fc1 activation (k-WTA=64, 25% of 256 units active during training)
   - Buffer logging optimization (per-class instead of per-sample, ~64x faster)
   - Forgetting sign fix (was inverted: old-new → new-old)

7. **Benchmark results (Split-MNIST, 1 epoch/task, 5 seeds):**

   | Config | Final ACC | BWT |
   |--------|-----------|-----|
   | naive | 0.1901 ± 0.0009 | -0.7855 ± 0.0004 |
   | EWC | 0.1901 ± 0.0012 | -0.7851 ± 0.0006 |
   | **two_stage_gate** | **0.7358 ± 0.0099** | **-0.1601 ± 0.0109** |

   Gate: **3.87x higher ACC**, **79.6% less forgetting** vs naive.

8. **Per-task retention (gate):**
   - Task 0 (digits 0,1): 99.1% → 90.2% (91% retention)
   - Task 1 (digits 2,3): 80.3% → 76.5% (95% retention)
   - Task 2 (digits 4,5): 89.9% → 51.8% (58% retention) ← weakest
   - Task 3 (digits 6,7): 97.1% → 67.9% (70% retention)
   - Task 4 (digits 8,9): current 81.5%

9. **Core-size sweep:** 10→50 showed linear improvement, 100 hit plateau for 2-class subtasks.

10. **Parameter tuning:**
    - `freq_threshold`: 50 → 500
    - `eps_gain`: 0.01 → 0.001
    - `eps_forget`: 0.05 → 0.50
    - `core_size_per_cluster`: 0 → 100
    - `k_wta`: 0 → 64

## What Remains (v1.2 / v2)

1. **Replay-only baseline.** Run naive + experience replay (no gate, no Fisher) to isolate gate's added value.

2. **Task-2 retention weakness.** Diagnose why digits 4,5 degrade more than other pairs.

3. **Multi-epoch testing.** Current benchmark uses 1 epoch/task for speed. 5-10 epochs would stress-test replay.

4. **Fast-weight layer (v2).** Add separate Hebbian adapter for moment-to-moment adaptation.

5. **Noisier benchmarks.** Permuted MNIST or Split-CIFAR10 to test with less clean task boundaries.

## Architecture Decisions Locked

- **Replay is the primary protection mechanism.** Fisher masks are supplementary, not sufficient.
- **Core-set buffer is necessary infrastructure.** FIFO alone cannot support replay.
- **EWC remains structurally insufficient.** Even with sparse fc1, EWC ≈ naive. Don't invest in it.
- **Phase 5 (MESU metaplasticity) is deferrable.** Sparse fc1 gains are marginal (+1.5% ACC).

## Quick Start

```bash
source .venv/bin/activate
python3 train.py --help
python3 train.py \
    --configs naive ewc two_stage_gate \
    --seeds 42 43 44 45 46 \
    --epochs 1 \
    --freq-threshold 500 \
    --eps-gain 0.001 \
    --eps-forget 0.50 \
    --core-size 100 \
    --k-wta 64 \
    --output-dir results/my_run
```

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Experiment harness, main training loop with ongoing replay |
| `gate.py` | `validate_and_commit` with interleaved replay, `is_candidate` Stage 1 filter |
| `buffer.py` | Episodic buffer with core-set protected entries |
| `model.py` | SlowCNN with optional sparse fc1 (k-WTA) |
| `baselines.py` | Naive and EWC training (EWC produces naive-identical results) |
| `findings/09_v1.1_benchmark_report.md` | Full benchmark report with per-task breakdown |

## Dependencies

Python 3.12 + uv. PyTorch 2.9.1 for ROCm 7.2.4 (AMD RX 7900 GRE). CPU suffices for MNIST-scale.
