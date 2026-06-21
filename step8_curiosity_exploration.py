"""Step 8: Curiosity-Driven Exploration — Novelty as Intrinsic Reward.

Uses the CA1 novelty signal (validated in Step 5) to drive exploration.
The agent chooses experiences, and the novelty of each experience is its
intrinsic reward. The agent learns to seek novel patterns and avoid familiar ones.

Design:
  The agent is presented with a stream of possible experiences (shape + position).
  It processes each through the autoencoder → Hopfield memory → CA1 novelty.
  Decision rule based on percentile-normalized novelty:
    - Novelty above 80th percentile of recent history → "study" (store in memory)
    - Novelty below 20th percentile → "explore elsewhere" (skip)
    - In between → "maybe study" with probability proportional to novelty

  The agent's memory grows over time as novel patterns are stored.
  As patterns become familiar, novelty drops → agent moves on.
  This drives coverage of the experience space.

Test:
  1. Train autoencoder on standard shapes (circle, square, triangle at center positions)
  2. Present a mixed stream: 20% familiar, 40% novel positions, 40% new shapes
  3. Does the agent preferentially store novel patterns over familiar ones?
  4. Does the agent explore diverse novelty types, or fixate on one?
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
from step5_novelty_detection import NoveltyDetector, generate_diamond, \
    rotate_image

OUT = Path('results/step8')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ExperienceSpace:
    """Generates experiences from different distributions."""

    def __init__(self):
        self.rng = np.random.RandomState(42)
        self.step = 0

    def sample(self) -> tuple:
        """Sample an experience: (image, label, novelty_type).

        Returns (image_tensor, novelty_type_string).
        novelty_type describes the kind of experience for analysis.
        """
        t = self.rng.uniform()
        img_size = 64

        if t < 0.20:  # Familiar (training distribution)
            s = self.rng.choice(SHAPE_TYPES)
            pos = self.rng.uniform(0.15, 0.85, 2)
            img, _ = generate_shape_image(
                size=img_size, shape=s, position=pos, radius=0.2,
                noise=0.01, seed=self.rng.randint(2**31)
            )
            label = 'familiar'
        elif t < 0.60:  # Novel position (extreme corners)
            s = self.rng.choice(['circle', 'square', 'triangle'])
            pos = self.rng.uniform(0.01, 0.08, 2) if self.rng.random() < 0.5 \
                  else self.rng.uniform(0.92, 0.99, 2)
            img, _ = generate_shape_image(
                size=img_size, shape=s, position=pos, radius=0.2,
                noise=0.01, seed=self.rng.randint(2**31)
            )
            label = 'novel_position'
        else:  # New shape (diamond)
            pos = self.rng.uniform(0.15, 0.85, 2)
            img, _ = generate_diamond(
                size=img_size, position=pos, radius=0.2,
                noise=0.01, seed=self.rng.randint(2**31)
            )
            label = 'new_shape'

        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        self.step += 1
        return img_t, label


class CuriosityDrivenAgent:
    """Agent that explores using CA1 novelty as intrinsic reward.

    Decision: based on percentile of current novelty vs recent history.
    """

    def __init__(self, model, sep, memory, novelty_window=200):
        self.model = model
        self.sep = sep
        self.memory = memory
        self.detector = NoveltyDetector()
        self.novelty_history = deque(maxlen=novelty_window)

    def process(self, image: torch.Tensor) -> dict:
        """Process an experience. Returns {novelty, is_novel, decision, stored}."""
        image = image.to(DEVICE)

        with torch.no_grad():
            h = self.model.encode(image)          # (1, 128)
            z = self.sep(h)                       # (1, 2000)
            h_retrieved, attn, logits = self.memory.retrieve(z, return_attention=True)

        # Compute novelty from attention distribution + key similarity
        if attn is not None:
            novelty = self.detector.compute_from_attention(
                attn, key_logits=logits).squeeze().item()
        else:
            novelty = 1.0  # Memory empty: everything is novel
        self.novelty_history.append(novelty)

        # Decision: compare to recent history (percentile-based)
        recent = np.array(list(self.novelty_history))
        if len(recent) > 10:
            p80 = np.percentile(recent, 80)
            p20 = np.percentile(recent, 20)
        else:
            p80, p20 = 0.1, 0.02

        if novelty > p80:
            decision = 'study'
            # Store without extra batch dim — store() expects (D,) not (1, D)
            self.memory.store_with_image(z.squeeze(0), h.squeeze(0), image.cpu())
            stored = True
        elif novelty > p20:
            # Moderate novelty: store with probability proportional to novelty
            prob = (novelty - p20) / max(p80 - p20, 1e-6)
            if np.random.random() < prob * 0.5:
                self.memory.store_with_image(z.squeeze(0), h.squeeze(0),
                                              image.cpu())
                stored = True
                decision = 'maybe_study'
            else:
                stored = False
                decision = 'skip'
        else:
            stored = False
            decision = 'skip'  # Low novelty: clearly familiar

        return {
            'novelty': novelty,
            'p80': p80,
            'p20': p20,
            'decision': decision,
            'stored': stored,
            'memory_size': self.memory.stored_count(),
        }


def visualize_exploration(agent, space, n_steps=200, save_path=None):
    """Run exploration and visualize which patterns the agent learns."""
    types_seen = {'familiar': 0, 'novel_position': 0, 'new_shape': 0}
    types_stored = {'familiar': 0, 'novel_position': 0, 'new_shape': 0}
    novelty_over_time = []
    decisions = []

    for step in range(n_steps):
        img, label = space.sample()
        result = agent.process(img)
        types_seen[label] = types_seen.get(label, 0) + 1
        if result['stored']:
            types_stored[label] = types_stored.get(label, 0) + 1
        novelty_over_time.append(result['novelty'])
        decisions.append(result['decision'])

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Novelty over time
    axes[0, 0].plot(novelty_over_time, alpha=0.7, linewidth=0.8)
    axes[0, 0].axhline(y=np.percentile(novelty_over_time, 80),
                        color='green', linestyle='--', alpha=0.5,
                        label='80th %ile (study threshold)')
    axes[0, 0].axhline(y=np.percentile(novelty_over_time, 20),
                        color='red', linestyle=':', alpha=0.5,
                        label='20th %ile (skip threshold)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Novelty Score')
    axes[0, 0].set_title('Novelty Over Time')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Decisions
    decision_counts = {d: decisions.count(d) for d in set(decisions)}
    axes[0, 1].bar(decision_counts.keys(), decision_counts.values(),
                   color=['blue', 'green', 'red'])

    axes[0, 1].set_title(f'Decisions ({n_steps} steps)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(alpha=0.3)

    # Seen vs stored by type
    x = np.arange(len(types_seen))
    width = 0.35
    types_list = sorted(types_seen.keys())
    seen_counts = [types_seen[t] for t in types_list]
    stored_counts = [types_stored[t] for t in types_list]
    axes[1, 0].bar(x - width / 2, seen_counts, width, label='Seen',
                   color='lightblue')
    axes[1, 0].bar(x + width / 2, stored_counts, width, label='Stored',
                   color='darkblue')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(types_list, rotation=15)
    axes[1, 0].set_title('Seen vs Stored by Pattern Type')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Storage rate by type
    storage_rate = [
        stored_counts[i] / max(seen_counts[i], 1) * 100
        for i in range(len(types_list))
    ]
    colors = ['green' if types_list[i] == 'familiar' else
              'orange' if types_list[i] == 'novel_position' else 'red'
              for i in range(len(types_list))]
    axes[1, 1].bar(types_list, storage_rate, color=colors)
    axes[1, 1].set_title('Storage Rate by Pattern Type')
    axes[1, 1].set_ylabel('Storage Rate (%)')
    for i, v in enumerate(storage_rate):
        axes[1, 1].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()

    return types_seen, types_stored, novelty_over_time, decisions


def main():
    print(f"Device: {DEVICE}\n")

    # Initialize components (same architecture as Steps 3-5)
    model = SparseAutoencoder(feature_dim=128).to(DEVICE)
    sep = PatternSeparator(128, 2000, 0.02)
    memory = TripleMemory(2000, 128, beta=2.0).to(DEVICE)

    # Train autoencoder on standard shapes (no extreme positions or diamonds)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 2001):
        imgs, _ = generate_batch(batch_size=64)
        imgs = imgs.to(DEVICE)
        recon, h, recon_loss, sp_loss = model(imgs)
        loss = recon_loss + sp_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Initial memory: {memory.stored_count()} patterns\n")

    agent = CuriosityDrivenAgent(model, sep, memory)
    space = ExperienceSpace()

    # ── Phase 1: Pre-fill memory with familiar patterns ──────────────
    print("=" * 60)
    print("Phase 1: Pre-fill memory with familiar experiences\n")

    for _ in range(200):
        img, label = space.sample()
        if label == 'familiar':
            agent.process(img)

    print(f"  Memory now has {memory.stored_count()} patterns\n")

    # ── Phase 2: Curiosity-driven exploration ────────────────────────
    print("=" * 60)
    print("Phase 2: Curiosity-driven exploration\n")
    print(f"  Stream: 20% familiar, 40% novel position, 40% new shape\n")

    seen, stored, novelties, decisions = visualize_exploration(
        agent, space, n_steps=500, save_path=OUT / 'exploration.png'
    )

    # Print summary
    print("\n  Exploration Summary:")
    print(f"  {'Type':<20} {'Seen':>6} {'Stored':>6} {'Rate':>8}")
    print(f"  {'-'*42}")
    types_list = sorted(seen.keys())
    for t in types_list:
        rate = stored[t] / max(seen[t], 1) * 100
        print(f"  {t:<20} {seen[t]:>6} {stored[t]:>6} {rate:>7.0f}%")

    total_seen = sum(seen.values())
    total_stored = sum(stored.values())
    print(f"  {'-'*42}")
    print(f"  {'Total':<20} {total_seen:>6} {total_stored:>6} "
          f"{total_stored/max(total_seen,1)*100:>7.0f}%")

    final_size = memory.stored_count()
    print(f"\n  Final memory size: {final_size} patterns")

    # Novelty signal statistics
    recent = novelties[-100:]
    print(f"\n  Novelty signal (last 100 steps):")
    print(f"    Mean: {np.mean(recent):.4f}")
    print(f"    Std:  {np.std(recent):.4f}")
    print(f"    P20:  {np.percentile(recent, 20):.4f}")
    print(f"    P80:  {np.percentile(recent, 80):.4f}")

    print(f"\nDone. Results in {OUT.resolve()}")


if __name__ == '__main__':
    main()
