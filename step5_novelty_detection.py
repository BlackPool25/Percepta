"""Step 5: CA1 Novelty Detection — Distinguishing Familiar vs. Novel Patterns.

In the brain:
  CA1 receives input from CA3 (retrieved pattern) and from EC (actual sensory input).
  CA1 compares the two. If they match -> familiar. If they don't -> novelty signal.
  This novelty signal modulates plasticity (neuromodulation).

In our architecture:
  Encoder produces h (features) from input.
  Memory retrieves h* (stored pattern matching the input's sparse key).
  CA1: novelty = distance(h, h*)
    - Low novelty: input matches a stored pattern (familiar)
    - High novelty: input is new or significantly different (novel)

Test:
  1. Train autoencoder on shapes
  2. Fill memory with familiar patterns
  3. Present both familiar and novel patterns
  4. Measure: does novelty score distinguish them?
  
Novel patterns: shapes at unseen positions, new shape types, rotated shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from hopfield_memory import PatternSeparator
from step2_memory_with_correlation import KeyValueHopfieldMemory
from step3b_sparse_autoencoder import SparseAutoencoder, generate_batch, \
    corrupt_sparse_key, SHAPE_TYPES, generate_shape_image
from step4_sleep_consolidation import TripleMemory

OUT = Path('results/step5')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NoveltyDetector:
    """CA1 analogue: computes novelty from the difference between
    actual features h and memory-retrieved features h*.

    novelty = 1 - cosine_similarity(h, h*)

    When h* closely matches h -> low novelty (familiar).
    When h* differs from h -> high novelty (novel/surprising).
    """

    def __init__(self, threshold: float = 0.3, window: int = 5):
        self.threshold = threshold
        self.recent_novelties = []

    def compute(self, h_actual: torch.Tensor,
                h_retrieved: torch.Tensor) -> torch.Tensor:
        """Legacy: compute novelty from feature-space distance.

        Retained for backward compatibility but deprecated for the
        attention-based novelty in compute_from_attention.
        """
        h_a = h_actual / (h_actual.norm(dim=1, keepdim=True) + 1e-8)
        h_r = h_retrieved / (h_retrieved.norm(dim=1, keepdim=True) + 1e-8)
        cosim = (h_a * h_r).sum(dim=1)
        novelty = 1.0 - cosim
        return novelty.clamp(0, 2)

    def compute_from_attention(self, attention_weights: torch.Tensor,
                                 key_logits: torch.Tensor | None = None
                                 ) -> torch.Tensor:
        """Compute novelty from attention distribution.

        Combines two signals:
        1. Attention concentration: max(softmax weight) — how focused is the match?
        2. Key similarity: max(logit) / max_possible — how close is the best match?

        When memory has N stored patterns:
        - One perfect match: both signals high → novelty ≈ 0
        - Many patterns blend: attn_confidence << 1.0 → novelty ≈ 1 - attn_confidence
        - Few patterns, weak match: attn_confidence ≈ 1.0 but key_similarity low → novelty high
        
        Returns: (B,) novelty scores in [0, 1]
        """
        max_weights = attention_weights.max(dim=1).values

        if key_logits is not None:
            # Normalize key logits to [0, 1] via sigmoid-like scaling
            # beta is the temperature; high logit = good match
            # A logit of 0 means cosine_similarity = 0, logit > 0 means
            # better-than-random match
            max_logits = key_logits.max(dim=1).values
            key_sim = torch.sigmoid(max_logits)
            novelty = 1.0 - max_weights * key_sim
        else:
            novelty = 1.0 - max_weights

        return novelty.clamp(0, 1)

    def is_novel(self, novelty: torch.Tensor,
                 baseline_mean: float = 0.0,
                 baseline_std: float = 1e-6,
                 z_threshold: float = 3.0) -> torch.Tensor:
        z = (novelty - baseline_mean) / max(baseline_std, 1e-8)
        return z > z_threshold

    def store_novelty(self, novelty: float):
        self.recent_novelties.append(novelty)
        if len(self.recent_novelties) > 100:
            self.recent_novelties.pop(0)


def generate_diamond(size=64, position=None, radius=0.2, noise=0.01, seed=None):
    """Generate a diamond (rotated square) shape."""
    rng = np.random.RandomState(seed)
    if position is None:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
    else:
        x, y = position
    img = np.zeros((size, size), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    # Diamond: |x - cx| + |y - cy| < r
    mask = (np.abs(xs - x) + np.abs(ys - y)) < radius
    img[mask] = 1.0
    if noise > 0:
        img += rng.randn(*img.shape) * noise
    return np.clip(img, 0.0, 1.0), -1


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D image by angle_deg degrees using affine transform.

    Uses bilinear-like interpolation via simple coordinate mapping.
    """
    h, w = img.shape
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy = w / 2, h / 2

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # Translate to origin, rotate, translate back
    x_shift = xs - cx
    y_shift = ys - cy
    x_rot = x_shift * cos_a - y_shift * sin_a + cx
    y_rot = x_shift * sin_a + y_shift * cos_a + cy

    # Clip to bounds
    x_rot = np.clip(x_rot, 0, w - 1).astype(int)
    y_rot = np.clip(y_rot, 0, h - 1).astype(int)

    rotated = np.zeros_like(img)
    rotated[ys, xs] = img[y_rot, x_rot]
    return rotated


def generate_novel_patterns(batch_size=32, novelty_type='unseen_position',
                            img_size=64, shape_filter=None):
    """Generate patterns that differ from the training distribution.

    novelty_type:
      'unseen_position': shapes at extreme corner positions
      'new_shape': diamond (not in training set)
      'rotated': shapes at unseen rotations (normal positions)
    shape_filter: if set, only generate this shape type (e.g., 'circle')
    """
    imgs = []
    labels = []
    rng = np.random.RandomState(42)

    for _ in range(batch_size):
        if novelty_type == 'unseen_position':
            s = shape_filter if shape_filter else rng.choice(SHAPE_TYPES)
            pos = rng.uniform(0.01, 0.08, 2) if rng.random() < 0.5 \
                  else rng.uniform(0.92, 0.99, 2)
            img, lbl = generate_shape_image(
                size=img_size, shape=s, position=pos,
                radius=0.2, noise=0.01, seed=rng.randint(2**31)
            )
        elif novelty_type == 'new_shape':
            pos = rng.uniform(0.15, 0.85, 2)
            img, lbl = generate_diamond(
                size=img_size, position=pos, radius=0.2,
                noise=0.01, seed=rng.randint(2**31)
            )
        elif novelty_type == 'rotated':
            s = shape_filter if shape_filter else rng.choice(SHAPE_TYPES)
            pos = rng.uniform(0.25, 0.75, 2)
            img, lbl = generate_shape_image(
                size=img_size, shape=s, position=pos,
                radius=0.2, noise=0.0, seed=rng.randint(2**31)
            )
            angle = rng.uniform(30, 330)
            img = rotate_image(img, angle)
            # Add noise after rotation
            img += rng.randn(*img.shape) * 0.01
            img = np.clip(img, 0.0, 1.0)
        else:
            s = rng.choice(SHAPE_TYPES)
            pos = rng.uniform(0.15, 0.85, 2)
            img, lbl = generate_shape_image(
                size=img_size, shape=s, position=pos,
                radius=0.2, noise=0.01, seed=rng.randint(2**31)
            )
        imgs.append(img)
        labels.append(lbl)

    imgs_t = torch.from_numpy(np.stack(imgs)).unsqueeze(1).float()
    return imgs_t, torch.tensor(labels)


def test_novelty_discrimination(model, memory, sep, detector, device=DEVICE):
    """Test: do familiar and novel patterns produce different novelty scores?

    Returns dict mapping category names to numpy arrays of novelty scores.
    Rotated category is further broken down by shape type.
    """
    results = {}

    family_imgs, _ = generate_batch(batch_size=64)
    family_imgs = family_imgs.to(device)

    novel_types = ['unseen_position', 'new_shape']
    novel_sets = {}
    for nt in novel_types:
        imgs, _ = generate_novel_patterns(batch_size=32, novelty_type=nt)
        novel_sets[nt] = imgs.to(device)

    # Rotated broken down by shape
    for shape in SHAPE_TYPES:
        imgs, _ = generate_novel_patterns(batch_size=32, novelty_type='rotated',
                                           shape_filter=shape)
        novel_sets[f'rotated_{shape}'] = imgs.to(device)

    with torch.no_grad():
        fam_mem = TripleMemory(memory.key_dim, memory.value_dim,
                               beta=2.0).to(device)
        h_fam = model.encode(family_imgs)
        for i in range(64):
            z = sep(h_fam[i].unsqueeze(0)).squeeze(0)
            fam_mem.store_with_image(z, h_fam[i], family_imgs[i])

        # Familiar
        h_fam_actual = model.encode(family_imgs)
        z_fam = torch.stack([
            sep(h_fam_actual[i].unsqueeze(0)).squeeze(0) for i in range(64)
        ])
        h_fam_retrieved = fam_mem.retrieve(z_fam)
        nov_fam = detector.compute(h_fam_actual, h_fam_retrieved)
        results['familiar'] = nov_fam.cpu().numpy()

        # Novel categories
        for ntype, n_imgs in novel_sets.items():
            h_n = model.encode(n_imgs)
            z_n = torch.stack([
                sep(h_n[i].unsqueeze(0)).squeeze(0) for i in range(h_n.size(0))
            ])
            h_n_retrieved = fam_mem.retrieve(z_n)
            nov_n = detector.compute(h_n, h_n_retrieved)
            results[ntype] = nov_n.cpu().numpy()

    return results


def train_model_and_fill_memory(n_steps=2000, device=DEVICE):
    """Train autoencoder and fill memory with diverse patterns."""
    model = SparseAutoencoder(feature_dim=128).to(device)
    sep = PatternSeparator(128, 2000, 0.02)
    memory = TripleMemory(2000, 128, beta=2.0).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, n_steps + 1):
        imgs, _ = generate_batch(batch_size=64)
        imgs = imgs.to(device)
        recon, h, recon_loss, sp_loss = model(imgs)
        loss = recon_loss + sp_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(8):
        imgs, _ = generate_batch(batch_size=64)
        imgs = imgs.to(device)
        with torch.no_grad():
            h = model.encode(imgs)
        for i in range(64):
            z = sep(h[i].unsqueeze(0)).squeeze(0)
            memory.store_with_image(z, h[i], imgs[i])

    return model, sep, memory


def main():
    print(f"Device: {DEVICE}\n")

    model, sep, memory = train_model_and_fill_memory(2000, DEVICE)
    print(f"Stored {memory.stored_count()} patterns in memory\n")

    detector = NoveltyDetector()

    scores = test_novelty_discrimination(model, memory, sep, detector, DEVICE)

    # Compute adaptive thresholds from familiar distribution
    fam_mean = scores['familiar'].mean()
    fam_std = max(scores['familiar'].std(), 1e-4)  # floor to prevent div-by-zero
    print(f"Baseline (familiar): mean={fam_mean:.6f} std_floor={fam_std:.6f}\n")

    print("=" * 60)
    print("Novelty Discrimination Results\n")

    for name in sorted(scores.keys()):
        ns = scores[name]
        nr = (ns.mean() - fam_mean) / fam_std
        print(f"  {name:>20}: mean={ns.mean():.4f} std={ns.std():.4f} "
              f"ratio={nr:.0f}x")

    # Plot grouped: familiar + novel_types + rotated_by_shape
    categories = ['familiar', 'unseen_position', 'new_shape']
    rotated_shapes = [k for k in sorted(scores.keys()) if k.startswith('rotated_')]
    categories += rotated_shapes

    plt.figure(figsize=(12, 6))
    positions = list(range(len(categories)))
    means = [scores[c].mean() for c in categories]
    stds = [scores[c].std() for c in categories]
    colors = ['green'] + ['red'] + ['orange'] + ['blue', 'purple', 'cyan']

    plt.bar(positions, means, yerr=stds, color=colors[:len(categories)],
            alpha=0.7, capsize=5)
    z3 = fam_mean + 3 * fam_std
    plt.axhline(y=z3, color='gray', linestyle='--',
                label=f'z=3 threshold ({z3:.4f})')
    plt.xticks(positions, categories, rotation=20)
    plt.ylabel('Novelty Score (1 - cosine similarity)')
    plt.title('CA1 Novelty: Per-Category Breakdown with Rotated-by-Shape')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT / 'novelty_by_category.png'
    plt.savefig(path, dpi=150)
    print(f"\n  Saved: {path}")

    print(f"\nDone. Results in {OUT.resolve()}")


if __name__ == '__main__':
    main()
