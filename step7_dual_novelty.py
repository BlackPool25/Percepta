"""Dual Novelty Signal: Fast (attention+key-similarity) + Slow (reconstruction error).

Extends Step 7's streaming setup with separate fast and slow novelty signals.
Tests the four-case interaction matrix and combined intrinsic reward.

Signals:
  Fast novelty  (attention+key-similarity): "Is this experience in memory?"
  Slow novelty  (reconstruction error):     "Does the slow system understand this?"

Four cases:
  Both high:       store AND train hard (genuinely new)
  Fast high, slow low:  store, light train (novel episode, weights cover it)
  Fast low, slow high:  don't store, train hard (familiar but not understood)
  Both low:        skip (routine)

Combined intrinsic reward = alpha * fast + (1-alpha) * slow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from hopfield_memory import PatternSeparator
from step2_memory_with_correlation import KeyValueHopfieldMemory
from step3b_sparse_autoencoder import SparseAutoencoder, generate_batch, \
    generate_shape_image, SHAPE_TYPES
from step4_sleep_consolidation import TripleMemory
from step5_novelty_detection import NoveltyDetector, generate_diamond
from step7_online_streaming import OnlineStream, BoundedMemory, DEVICE, OUT

OUT = Path('results/step7_dual')
OUT.mkdir(parents=True, exist_ok=True)


class DualNoveltyAgent:
    """Agent with separate fast and slow novelty signals.

    Fast novelty:  rate of change of reconstruction error
                   "Is this a novel event/episode boundary?"
                   Gates memory storage (event boundaries get stored)

    Slow novelty:  absolute reconstruction error
                   "Does the slow system understand this?"
                   Gates learning rate and sleep priority

    Combined intrinsic reward = alpha * fast + (1-alpha) * slow
    """

    def __init__(self, model, sep, memory, alpha=0.5, lr=1e-4,
                 update_interval=32, replay_window=128):
        self.model = model
        self.sep = sep
        self.memory = memory
        self.detector = NoveltyDetector()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.alpha = alpha
        self.update_interval = update_interval
        self.replay_window = replay_window
        self.step = 0
        self.combined_history = deque(maxlen=500)
        self.fast_history = deque(maxlen=500)
        self.slow_history = deque(maxlen=500)
        self.recent_inputs = deque(maxlen=replay_window)

    def _fast_novelty(self, z: torch.Tensor) -> float:
        """Fast novelty: hamming distance to the nearest stored key.

        For each stored key, computes the number of shared active bits.
        Fast novelty = 1 - (max overlap / k_active).

        0.0 = perfect match (all active bits match a stored pattern)
        1.0 = no overlap (no stored pattern shares any active bit)

        This avoids the blending problem by only looking at the nearest
        stored pattern, not a weighted combination.
        """
        if self.memory.stored_count() == 0:
            return 1.0

        k_active = z.sum().item()
        if k_active == 0:
            return 1.0

        Z = torch.stack(self.memory.keys, dim=0).to(z.device)
        # Overlap = (z_bool & Z_bool).sum() but & doesn't work on Float
        overlaps = (z.unsqueeze(0) * Z).sum(dim=1)
        max_overlap = overlaps.max().item()
        return 1.0 - max_overlap / k_active

    def process(self, image: torch.Tensor) -> dict:
        self.step += 1
        image = image.to(DEVICE)

        with torch.no_grad():
            h = self.model.encode(image)
            z = self.sep(h)
            recon = self.model.decode(h)

        # ── Fast novelty: hamming distance to nearest stored key ──────
        # "Is this input different from what's already in memory?"
        # Hard nearest neighbor avoids the blending problem
        slow_novelty = F.mse_loss(recon, image).item()
        fast_novelty = self._fast_novelty(z.squeeze(0))

        # ── Combined intrinsic reward ───────────────────────────────────
        combined = self.alpha * fast_novelty + (1 - self.alpha) * slow_novelty
        self.combined_history.append(combined)
        self.fast_history.append(fast_novelty)
        self.slow_history.append(slow_novelty)

        # ── Decisions (separate thresholds, OR logic) ──────────────────
        # Storage triggers if EITHER signal exceeds its own threshold.
        # This prevents a broken fast signal from vetoing storage when
        # the slow signal is high, and vice versa.
        recent_fast = np.array(self.fast_history)
        recent_slow = np.array(self.slow_history)
        n = len(recent_fast)

        fast_threshold = np.percentile(recent_fast, 70) if n > 20 else 0.5
        slow_threshold = np.percentile(recent_slow, 70) if n > 20 else 0.2

        store = (fast_novelty > fast_threshold) or (slow_novelty > slow_threshold)
        if store:
            self.memory.store_with_image(
                z.squeeze(0), h.squeeze(0), image.cpu())

        # Slow novelty → training intensity (applied in _online_update)
        self.recent_inputs.append((image.cpu(), h.cpu(), z.cpu(), slow_novelty))

        # Classify into the four cases
        fast_high = fast_novelty > fast_threshold
        slow_high = slow_novelty > slow_threshold

        if fast_high and slow_high:
            case = 'both_high'
        elif fast_high and not slow_high:
            case = 'fast_high'
        elif not fast_high and slow_high:
            case = 'slow_high'
        else:
            case = 'both_low'

        return {
            'fast_novelty': fast_novelty,
            'slow_novelty': slow_novelty,
            'combined': combined,
            'case': case,
            'fast_threshold': fast_threshold,
            'slow_threshold': slow_threshold,
            'memory_size': self.memory.stored_count(),
            'stored': store,
        }

    def _online_update(self):
        """Train autoencoder on recent experiences."""
        batch = list(self.recent_inputs)
        imgs = torch.cat([b[0] for b in batch[-self.update_interval:]]).to(DEVICE)

        recon, h, recon_loss, sp_loss = self.model(imgs)
        loss = recon_loss + sp_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    print(f"Device: {DEVICE}\n")

    model = SparseAutoencoder(feature_dim=128).to(DEVICE)
    sep = PatternSeparator(128, 2000, 0.02)
    memory = BoundedMemory(2000, 128, beta=2.0, max_size=200).to(DEVICE)
    agent = DualNoveltyAgent(model, sep, memory, alpha=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    stream = OnlineStream(total_steps=5000)

    # Tracking
    fast_novs, slow_novs, combineds = [], [], []
    cases = []
    memory_sizes = []
    stored_flags = []

    print("=" * 60)
    print("Dual Novelty Streaming (alpha=0.5)")
    print("Phase schedule:")
    print("   0-500:   Center positions (familiar)")
    print("  500-1500: Extreme positions (novel)")
    print(" 1500-2500: Center + diamonds")
    print(" 2500-3500: Mixed all types")
    print(" 3500-4500: Back to center")
    print(" 4500-5000: Extreme again\n")

    for step, (img_np, label, phase) in enumerate(stream):
        img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()
        result = agent.process(img_t)

        fast_novs.append(result['fast_novelty'])
        slow_novs.append(result['slow_novelty'])
        combineds.append(result['combined'])
        cases.append(result['case'])
        memory_sizes.append(result['memory_size'])
        stored_flags.append(result['stored'])

        if step % 500 == 0 or step == 0:
            print(f"  step={step:5d} phase={phase:>15} "
                  f"fast={result['fast_novelty']:.4f} "
                  f"slow={result['slow_novelty']:.4f} "
                  f"combined={result['combined']:.4f} "
                  f"mem={result['memory_size']:3d} "
                  f"case={result['case']}")

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n  Final memory: {memory.stored_count()} patterns")

    # Case distribution
    case_counts = {}
    for c in cases:
        case_counts[c] = case_counts.get(c, 0) + 1
    print(f"\n  Case distribution (alpha={agent.alpha}):")
    for c in ['both_high', 'fast_high', 'slow_high', 'both_low']:
        pct = case_counts.get(c, 0) / len(cases) * 100
        print(f"    {c:>15}: {case_counts.get(c, 0):5d} ({pct:.1f}%)")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # 1. Fast vs slow novelty over time
    axes[0].plot(fast_novs, alpha=0.5, linewidth=0.5, label='Fast (attention)')
    axes[0].plot(slow_novs, alpha=0.5, linewidth=0.5, label='Slow (recon error)')
    axes[0].plot(combineds, alpha=0.7, linewidth=0.5, label='Combined')
    axes[0].set_ylabel('Novelty')
    axes[0].set_title('Dual Novelty Signals Over Time (alpha=0.5)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 2. Case distribution over time
    case_colors = {
        'both_high': 'darkred', 'fast_high': 'orange',
        'slow_high': 'blue', 'both_low': 'gray',
    }
    case_indices = {
        'both_high': 3, 'fast_high': 2, 'slow_high': 1, 'both_low': 0,
    }
    case_over_time = [case_indices.get(c, 0) for c in cases]
    axes[1].scatter(range(len(cases)), case_over_time,
                    c=[case_colors.get(c, 'gray') for c in cases],
                    s=2, alpha=0.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['both_low', 'slow_high', 'fast_high', 'both_high'])
    axes[1].set_ylabel('Case')
    axes[1].set_title('Novelty Case per Step')
    axes[1].grid(alpha=0.3)

    # 3. Memory growth
    axes[2].plot(memory_sizes, linewidth=1)
    axes[2].set_ylabel('Memory Size')
    axes[2].set_title('Memory Growth')
    axes[2].grid(alpha=0.3)

    # 4. Storage rate + cases (100-step rolling)
    window = 100
    storage_rate = np.convolve(stored_flags, np.ones(window) / window, mode='same')
    case_rate_high = np.convolve(
        [1 if c == 'both_high' else 0 for c in cases],
        np.ones(window) / window, mode='same')
    case_rate_low = np.convolve(
        [1 if c == 'both_low' else 0 for c in cases],
        np.ones(window) / window, mode='same')
    axes[3].plot(storage_rate, linewidth=1, label='Storage rate')
    axes[3].plot(case_rate_high, linewidth=0.8, alpha=0.7,
                 label='Rate both_high')
    axes[3].plot(case_rate_low, linewidth=0.8, alpha=0.7,
                 label='Rate both_low')
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Rate (smoothed)')
    axes[3].set_title('Storage and Case Rates')
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    path = OUT / 'dual_novelty.png'
    plt.savefig(path, dpi=150)
    print(f"\n  Saved: {path}")

    # Print mean novelty by phase
    print("\n  Mean novelty by phase:")
    phase_boundaries = [0, 500, 1500, 2500, 3500, 4500, 5000]
    phase_names = ['center', 'extreme', 'mixed+new', 'mixed_all', 'back', 'extreme_again']
    print(f"  {'Phase':<15} {'Fast':>8} {'Slow':>8} {'Combined':>10} {'Store':>8}")
    print(f"  {'-'*51}")
    for i, (p_start, p_end, p_name) in enumerate(
        zip(phase_boundaries[:-1], phase_boundaries[1:], phase_names)):
        seg_fast = np.mean(fast_novs[p_start:p_end])
        seg_slow = np.mean(slow_novs[p_start:p_end])
        seg_comb = np.mean(combineds[p_start:p_end])
        seg_store = np.mean(stored_flags[p_start:p_end]) * 100
        print(f"  {p_name:<15} {seg_fast:>8.4f} {seg_slow:>8.4f} "
              f"{seg_comb:>10.4f} {seg_store:>7.1f}%")

    print(f"\nDone. Results in {OUT.resolve()}")


if __name__ == '__main__':
    main()
