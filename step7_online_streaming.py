"""Step 7: Online Streaming — Single-Pass, Non-IID, No Epochs.

Tests whether the validated architecture works in a continuous stream
where each experience arrives one at a time, with distribution shifts
and no task boundaries.

Key design changes from batch mode:
  1. Each experience is processed individually (no batching assumptions)
  2. Autoencoder updates on sliding windows (last N experiences, no shuffle)
  3. Memory stores incrementally (one pattern at a time)
  4. Novelty threshold is adaptive (percentile-based on recent history)
  5. Distribution shifts and concept drift are expected

Test protocol:
  Stream of 5000 individual shape images with phased distribution shifts:
    Phase 0 (0-500):    Familiar shapes at center positions
    Phase 1 (500-1500):  Shift to novel positions (extreme corners)
    Phase 2 (1500-2500): Shift back to center + new shapes (diamonds)
    Phase 3 (2500-3500): Mixed distribution (all types)
    Phase 4 (3500-4500): Return to initial distribution (familiar only)
    Phase 5 (4500-5000): Novel positions again

  Key questions:
    - Does the autoencoder retain knowledge across distribution shifts?
    - Does the memory fill with diverse patterns or collapse to one type?
    - Does novelty correctly rise during shifts and fall during stability?
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

OUT = Path('results/step7')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class OnlineStream:
    """Continuous stream of shape images with distribution shifts."""

    def __init__(self, total_steps=5000):
        self.total_steps = total_steps
        self.step = 0
        self.rng = np.random.RandomState(42)

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= self.total_steps:
            raise StopIteration
        s = self.step
        self.step += 1

        # Determine current phase distribution
        if s < 500:
            phase = 'center'  # Familiar: center positions
        elif s < 1500:
            phase = 'extreme'  # Novel: extreme positions
        elif s < 2500:
            phase = 'mixed_center'  # Center + diamonds
        elif s < 3500:
            phase = 'mixed_all'  # All types mixed
        elif s < 4500:
            phase = 'back_to_center'  # Return to initial
        else:
            phase = 'extreme_again'  # Novel again

        img, label = self._generate(phase)
        return img, label, phase

    def _generate(self, phase):
        img_size = 64
        if phase == 'center':
            s = self.rng.choice(SHAPE_TYPES)
            pos = self.rng.uniform(0.2, 0.8, 2)
            img, lbl = generate_shape_image(
                size=img_size, shape=s, position=pos, radius=0.2,
                noise=0.01, seed=self.rng.randint(2**31))
            return img, 'familiar'
        elif phase == 'extreme' or phase == 'extreme_again':
            s = self.rng.choice(['circle', 'square', 'triangle'])
            pos = self.rng.uniform(0.01, 0.08, 2) if self.rng.random() < 0.5 \
                  else self.rng.uniform(0.92, 0.99, 2)
            img, lbl = generate_shape_image(
                size=img_size, shape=s, position=pos, radius=0.2,
                noise=0.01, seed=self.rng.randint(2**31))
            return img, 'novel_position'
        elif phase == 'mixed_center':
            if self.rng.random() < 0.5:
                s = self.rng.choice(SHAPE_TYPES)
                pos = self.rng.uniform(0.2, 0.8, 2)
                img, lbl = generate_shape_image(
                    size=img_size, shape=s, position=pos, radius=0.2,
                    noise=0.01, seed=self.rng.randint(2**31))
                return img, 'familiar'
            else:
                pos = self.rng.uniform(0.2, 0.8, 2)
                img, lbl = generate_diamond(
                    size=img_size, position=pos, radius=0.2,
                    noise=0.01, seed=self.rng.randint(2**31))
                return img, 'new_shape'
        elif phase == 'mixed_all':
            t = self.rng.uniform()
            if t < 0.33:
                return self._generate('center')
            elif t < 0.66:
                return self._generate('extreme')
            else:
                return self._generate('mixed_center')
        elif phase == 'back_to_center':
            return self._generate('center')


class BoundedMemory(TripleMemory):
    """Memory with max capacity. Evicts oldest patterns when full.

    Prevents the blending problem: limited coverage means novel patterns
    remain detectable even after many steps.
    """

    def __init__(self, key_dim: int, value_dim: int, beta: float = 1.0,
                 max_size: int = 200):
        super().__init__(key_dim, value_dim, beta)
        self.max_size = max_size

    def store_with_image(self, key, value, image):
        if self.stored_count() >= self.max_size:
            self.keys.pop(0)
            self.values.pop(0)
            self.images.pop(0)
        super().store_with_image(key, value, image)


class OnlineAgent:
    """Processes a streaming experience one at a time."""

    def __init__(self, model, sep, memory, lr=1e-4,
                 update_interval=32, replay_window=128):
        self.model = model
        self.sep = sep
        self.memory = memory
        self.detector = NoveltyDetector()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.update_interval = update_interval
        self.replay_window = replay_window
        self.step = 0
        self.novelty_history = deque(maxlen=500)
        self.recent_inputs = deque(maxlen=replay_window)

    def process(self, image: torch.Tensor) -> dict:
        self.step += 1
        image = image.to(DEVICE)

        with torch.no_grad():
            h = self.model.encode(image)
            z = self.sep(h)
            recon = self.model.decode(h)

        # Novelty = reconstruction error (predictive coding)
        # High reconstruction error = model doesn't understand input = novel
        # Low reconstruction error = model has seen similar inputs = familiar
        # This avoids the blending problem entirely by not depending on memory
        reconstruction_error = F.mse_loss(recon, image).item()

        # Also compute memory retrieval via attention for storage decision
        h_retrieved, attn, logits = self.memory.retrieve(z, return_attention=True)
        if attn is not None and self.memory.stored_count() >= 5:
            attn_novelty = self.detector.compute_from_attention(
                attn, key_logits=logits).squeeze().item()
        elif attn is not None:
            attn_novelty = self.detector.compute(
                h, h_retrieved).squeeze().item()
        else:
            attn_novelty = 1.0

        # Use reconstruction error for the novelty signal (stable across scales)
        # Use attention novelty for storage decisions (what to keep in memory)
        novelty = reconstruction_error
        self.novelty_history.append(novelty)
        self.recent_inputs.append((image.cpu(), h.cpu(), z.cpu()))

        # Store if highly novel according to reconstruction error
        recent = np.array(list(self.novelty_history))
        threshold = np.percentile(recent, 70) if len(recent) > 20 else 0.01
        if novelty > threshold:
            self.memory.store_with_image(z.squeeze(0), h.squeeze(0), image.cpu())

        # Incremental autoencoder update on recent window
        if self.step % self.update_interval == 0 and len(self.recent_inputs) >= 8:
            self._online_update()

        return {
            'novelty': novelty,
            'threshold': threshold,
            'memory_size': self.memory.stored_count(),
            'phase': None,
        }

    def _online_update(self):
        """Train autoencoder on recent experiences (no shuffle)."""
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
    agent = OnlineAgent(model, sep, memory)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    stream = OnlineStream(total_steps=5000)

    # Held-out test sets (fixed, never trained on)
    held_out = {}
    rng_hold = np.random.RandomState(999)
    for label, shape, pos_gen in [
        ('center', 'circle', lambda: rng_hold.uniform(0.2, 0.8, 2)),
        ('extreme', 'square', lambda: np.array([0.97, 0.03]) if rng_hold.random() < 0.5 else np.array([0.03, 0.97])),
    ]:
        imgs = []
        for _ in range(50):
            img, _ = generate_shape_image(64, shape, pos_gen(), 0.2, 0.01, rng_hold.randint(2**31))
            imgs.append(img)
        held_out[label] = torch.from_numpy(np.stack(imgs)).unsqueeze(1).float().to(DEVICE)

    held_out_errors = {'center': [], 'extreme': [], 'step': []}

    def eval_held_out(current_step=0):
        model.eval()
        with torch.no_grad():
            for label in ['center', 'extreme']:
                recon, _, _, _ = model(held_out[label])
                held_out_errors[label].append(F.mse_loss(recon, held_out[label]).item())
        held_out_errors['step'].append(current_step)
        model.train()

    eval_held_out(0)

    # Tracking
    novelties = []
    thresholds = []
    memory_sizes = []
    phases = []
    recon_errors = []

    print("=" * 60)
    print("Streaming 5000 experiences (one at a time, no epochs)")
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

        novelties.append(result['novelty'])
        thresholds.append(result['threshold'])
        memory_sizes.append(result['memory_size'])
        phases.append(phase)

        if step % 500 == 0 and step > 0:
            eval_held_out(step)
            print(f"  step={step:5d} phase={phase:>15} "
                  f"novelty={result['novelty']:.4f} "
                  f"mem={result['memory_size']:4d} "
                  f"held_center={held_out_errors['center'][-1]:.4f} "
                  f"held_extreme={held_out_errors['extreme'][-1]:.4f}")
        elif step % 100 == 0 or step == 0:
            print(f"  step={step:5d} phase={phase:>15} "
                  f"novelty={result['novelty']:.4f} "
                  f"mem={result['memory_size']:4d}")

    print(f"\n  Final memory size: {memory.stored_count()} patterns")

    # ── Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 1. Novelty + threshold over time
    axes[0].plot(novelties, alpha=0.5, linewidth=0.5, label='Novelty')
    axes[0].plot(thresholds, alpha=0.7, linewidth=0.8,
                 label='Adaptive threshold (60th %ile)')
    axes[0].set_ylabel('Novelty')
    axes[0].set_title('Online Streaming: Novelty Signal Over Time')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 2. Memory growth
    axes[1].plot(memory_sizes, linewidth=1)
    axes[1].set_ylabel('Memory Size (patterns)')
    axes[1].set_title('Memory Growth')
    axes[1].grid(alpha=0.3)

    # 3. Phase transitions
    phase_colors = {
        'center': 'green', 'extreme': 'red', 'mixed_center': 'orange',
        'mixed_all': 'purple', 'back_to_center': 'blue',
        'extreme_again': 'pink',
    }
    phase_regions = []
    current_phase = phases[0]
    start = 0
    for i, p in enumerate(phases + [None]):
        if p != current_phase:
            phase_regions.append((start, i, current_phase))
            current_phase = p
            start = i

    for start, end, p in phase_regions:
        axes[2].axvspan(start, end, alpha=0.2, color=phase_colors.get(p, 'gray'))
    axes[2].set_ylabel('Phase')
    axes[2].set_title('Distribution Phases (colored regions)')
    axes[2].set_ylim(0, 1)
    # Legend
    for p, c in phase_colors.items():
        axes[2].plot([], [], color=c, label=p, linewidth=5)
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(alpha=0.3)

    # 4. Memory size derivative (storage rate over time)
    mem_array = np.array(memory_sizes)
    storage_rate = np.diff(mem_array, prepend=0)
    window = 100
    storage_rate_smooth = np.convolve(storage_rate,
                                       np.ones(window) / window, mode='same')
    axes[3].plot(storage_rate_smooth, linewidth=1)
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Storage Rate (patterns/step, smoothed)')
    axes[3].set_title('When Is the Agent Storing New Patterns?')
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    path = OUT / 'online_streaming.png'
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")

    # Final held-out eval
    eval_held_out(5000)

    # ── Held-out plot ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    steps_ho = held_out_errors['step']
    ax2.plot(steps_ho, held_out_errors['center'], 'go-', linewidth=2,
             label='Center (held-out)')
    ax2.plot(steps_ho, held_out_errors['extreme'], 'rs--', linewidth=2,
             label='Extreme (held-out)')
    ylim = ax2.get_ylim()
    for p_start, p_end, p_name in [
        (0, 500, 'center'), (500, 1500, 'extreme'), (1500, 2500, 'mixed+new'),
        (2500, 3500, 'mixed_all'), (3500, 4500, 'back'), (4500, 5000, 'extreme_again'),
    ]:
        ax2.axvspan(p_start, p_end, alpha=0.1, color='gray')
        ax2.text((p_start + p_end) / 2, ylim[1] * 0.95,
                 p_name, ha='center', fontsize=7, alpha=0.6)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reconstruction MSE (held-out)')
    ax2.set_title('Held-Out Reconstruction Error: Center vs Extreme')
    ax2.legend()
    ax2.grid(alpha=0.3)
    path2 = OUT / 'held_out_recon.png'
    plt.tight_layout()
    plt.savefig(path2, dpi=150)
    print(f"  Saved: {path2}")

    # Summary stats by phase
    print("\n  Per-Phase Novelty Stats:")
    print(f"  {'Phase':<20} {'Mean':>6} {'Std':>6} {'MemGrowth':>10}")
    print(f"  {'-'*44}")
    for p in ['center', 'extreme', 'mixed_center', 'mixed_all',
              'back_to_center', 'extreme_again']:
        phase_indices = [i for i, ph in enumerate(phases) if ph == p]
        if phase_indices:
            phase_nov = [novelties[i] for i in phase_indices]
            mem_growth = memory_sizes[phase_indices[-1]] - memory_sizes[phase_indices[0]]
            print(f"  {p:<20} {np.mean(phase_nov):>6.4f} "
                  f"{np.std(phase_nov):>6.4f} {mem_growth:>10}")

    print(f"\nDone. Results in {OUT.resolve()}")


if __name__ == '__main__':
    main()
