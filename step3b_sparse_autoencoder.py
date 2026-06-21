"""Step 3b: Sparse Autoencoder with Geometric Shapes + Hopfield Memory.

Validates: learned features → DG → Hopfield memory → retrieval → decode.

Pipeline:
  Synthetic shape image → Encoder → features h (128-dim)
    → DG pattern separation → sparse key z (40/2000 active)
      → Hopfield memory stores (z, h)
      → Retrieval from corrupted query: h* = softmax(β·z_q·Z^T)·V
        → Decoder(h*) → reconstructed image

Tests:
  1. Does the autoencoder avoid collapse? (reconstruction loss)
  2. Do learned features work with Hopfield memory? (retrieval cosim)
  3. Can we reconstruct shapes correctly after memory retrieval? (visual check)

Data: geometric shapes (circle, square, triangle) at random positions.
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

OUT = Path('results/step3b')
OUT.mkdir(parents=True, exist_ok=True)


# ─── Shape Image Generator ─────────────────────────────────────────────

SHAPE_TYPES = ['circle', 'square', 'triangle']


def generate_shape_image(size=64, shape='circle', position=None, radius=0.2,
                         noise=0.0, seed=None):
    """Generate a grayscale image with a geometric shape.

    Args:
        size: image width/height in pixels
        shape: 'circle', 'square', or 'triangle'
        position: (x, y) in [0,1] normalized coords (None = random)
        radius: shape size as fraction of image
        noise: Gaussian noise stddev to add

    Returns:
        image: (size, size) numpy array in [0, 1]
        label: shape index (0, 1, 2)
    """
    rng = np.random.RandomState(seed)

    if position is None:
        margin = radius + 0.05
        x = rng.uniform(margin, 1.0 - margin)
        y = rng.uniform(margin, 1.0 - margin)
    else:
        x, y = position

    img = np.zeros((size, size), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    # Normalized coords: x is column, y is row

    if shape == 'circle':
        mask = ((xs - x) ** 2 + (ys - y) ** 2) < radius ** 2
    elif shape == 'square':
        half = radius
        mask = (np.abs(xs - x) < half) & (np.abs(ys - y) < half)
    elif shape == 'triangle':
        # Equilateral triangle pointing up
        x0, y0 = x, y + radius
        x1, y1 = x - radius, y - radius
        x2, y2 = x + radius, y - radius
        # Barycentric check
        v0x, v0y = x2 - x1, y2 - y1
        v1x, v1y = x0 - x2, y0 - y2
        v2x, v2y = x1 - x0, y1 - y0
        dot00 = v0x * (xs - x1) + v0y * (ys - y1)
        dot01 = v0x * v0x + v0y * v0y
        dot02 = v1x * (xs - x2) + v1y * (ys - y2)
        dot11 = v1x * v1x + v1y * v1y
        dot20 = v2x * (xs - x0) + v2y * (ys - y0)
        dot21 = v2x * v2x + v2y * v2y
        mask = (dot00 >= 0) & (dot00 <= dot01) & \
               (dot02 >= 0) & (dot02 <= dot11) & \
               (dot20 >= 0) & (dot20 <= dot21)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    img[mask] = 1.0
    if noise > 0:
        img += rng.randn(*img.shape) * noise
    img = np.clip(img, 0.0, 1.0)

    label = SHAPE_TYPES.index(shape)
    return img, label


def generate_batch(batch_size=64, size=64, shape=None, position_noise=0.0,
                   noise=0.01):
    """Generate a batch of shape images with labels."""
    imgs = []
    labels = []
    rng = np.random.RandomState(None)

    for _ in range(batch_size):
        if shape is None:
            s = rng.choice(SHAPE_TYPES)
        else:
            s = shape
        pos = None
        if position_noise > 0:
            # Slightly perturb a random base position
            base = rng.uniform(0.2, 0.8, 2)
            pos = base + rng.randn(2) * position_noise
            pos = np.clip(pos, 0.1, 0.9)
        img, lbl = generate_shape_image(size=size, shape=s, position=pos,
                                         noise=noise, seed=rng.randint(2**31))
        imgs.append(img)
        labels.append(lbl)

    imgs = torch.from_numpy(np.stack(imgs)).unsqueeze(1).float()
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels


# ─── Sparse Autoencoder ─────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """CNN autoencoder with sparsity-regularized bottleneck.

    Encoder: 64×64 grayscale → features h (feature_dim)
    Decoder: features h → 64×64 grayscale reconstruction
    """

    def __init__(self, feature_dim=128, sparsity_weight=0.001,
                 img_size=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.sparsity_weight = sparsity_weight
        self.img_size = img_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),   # 32×32
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),   # 16×16
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),   # 8×8
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 4×4
            nn.ReLU(),
            nn.Flatten(),                                  # 128 * 4 * 4 = 2048
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2,
                               output_padding=1),  # 8×8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2,
                               output_padding=1),  # 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2,
                               output_padding=1),  # 32×32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 5, stride=2, padding=2,
                               output_padding=1),  # 64×64
            nn.Sigmoid(),
        )

    def encode(self, x):
        """x: (B, 1, H, W) → h: (B, feature_dim)"""
        return self.encoder(x)

    def decode(self, h):
        """h: (B, feature_dim) → reconstructed: (B, 1, H, W)"""
        return self.decoder(h)

    def forward(self, x):
        h = self.encode(x)
        recon = self.decode(h)

        # Sparsity loss: L1 penalty on features
        sparsity_loss = self.sparsity_weight * h.abs().mean()

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)

        return recon, h, recon_loss, sparsity_loss

    def get_features(self, x):
        with torch.no_grad():
            return self.encode(x)


# ─── Training ────────────────────────────────────────────────────────────

def train_autoencoder(
    model, n_steps=5000, batch_size=64, lr=1e-3,
    device='cpu', log_every=200,
):
    """Train sparse autoencoder on synthetic shapes."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(1, n_steps + 1):
        imgs, _ = generate_batch(batch_size=batch_size)
        imgs = imgs.to(device)

        recon, h, recon_loss, sparsity_loss = model(imgs)
        total_loss = recon_loss + sparsity_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        losses.append(total_loss.item())

        if step % log_every == 0 or step == 1:
            print(f"  step={step:5d} recon={recon_loss.item():.6f} "
                  f"sparse={sparsity_loss.item():.6f} "
                  f"h_mean={h.mean().item():.4f} h_std={h.std().item():.4f}")

    return losses


def evaluate_reconstruction(model, n_samples=10, device='cpu', title=''):
    """Visualize autoencoder reconstructions."""
    imgs, labels = generate_batch(batch_size=n_samples)
    imgs = imgs.to(device)
    with torch.no_grad():
        recon, h, _, _ = model(imgs)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    for i in range(n_samples):
        axes[0, i].imshow(imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Recon', fontsize=10)
    plt.suptitle(f'Autoencoder Reconstruction{title}')
    plt.tight_layout()
    path = OUT / f'reconstruction{title.replace(" ", "_")}.png'
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def corrupt_sparse_key(z: torch.Tensor, level: float = 0.5) -> torch.Tensor:
    """Zero out `level` fraction of active units in sparse key z."""
    z_c = z.clone()
    active = torch.where(z > 0.5)[0]
    if len(active) > 0:
        n = max(1, int(len(active) * level))
        idx = active[torch.randperm(len(active))[:n]]
        z_c[idx] = 0.0
    return z_c


def evaluate_memory_retrieval(model, memory, sep, n_samples=50, device='cpu',
                              corruption_level=0.5):
    """Test: store shape features, corrupt query key, retrieve, decode."""
    imgs, labels = generate_batch(batch_size=n_samples)
    imgs = imgs.to(device)

    with torch.no_grad():
        h = model.encode(imgs)

    # Store all in memory
    memory.reset()
    all_keys = []
    for i in range(n_samples):
        z = sep(h[i].unsqueeze(0)).squeeze(0)
        memory.store(z, h[i])
        all_keys.append(z)

    # Corrupt the SPARSE KEYS (pattern completion test)
    corrupted_keys = []
    for i in range(n_samples):
        z_q = corrupt_sparse_key(all_keys[i].to(device), level=corruption_level)
        corrupted_keys.append(z_q)

    # Retrieve via memory and decode
    retrieved_h = []
    for i in range(n_samples):
        h_r = memory.retrieve(corrupted_keys[i].unsqueeze(0)).squeeze(0)
        retrieved_h.append(h_r)
    retrieved_h = torch.stack(retrieved_h)

    # Also: retrieve WITHOUT corruption (baseline)
    direct_h = []
    for i in range(n_samples):
        h_d = memory.retrieve(all_keys[i].unsqueeze(0).to(device)).squeeze(0)
        direct_h.append(h_d)
    direct_h = torch.stack(direct_h)

    # Decode
    with torch.no_grad():
        recon_original = model.decode(h)
        recon_retrieved = model.decode(retrieved_h)
        recon_direct = model.decode(direct_h)

    # Metrics
    orig_mse = F.mse_loss(recon_original, imgs).item()
    corrupt_query_mse = F.mse_loss(h, retrieved_h).item()  # feature mse
    retrieved_mse = F.mse_loss(recon_retrieved, imgs).item()
    direct_mse = F.mse_loss(recon_direct, imgs).item()

    # Cosine similarity between retrieved and original features
    cosims = F.cosine_similarity(retrieved_h, h)
    mean_cosim = cosims.mean().item()

    return {
        'orig_mse': orig_mse,
        'direct_mse': direct_mse,
        'retrieved_mse': retrieved_mse,
        'feature_cosim': mean_cosim,
    }


def visualize_memory_test(model, memory, sep, n=8, device='cpu'):
    """Visualize: original → direct memory → corrupted key retrieval → decoded."""
    torch.manual_seed(42)
    imgs, labels = generate_batch(batch_size=n)
    imgs = imgs.to(device)

    with torch.no_grad():
        h = model.encode(imgs)

    memory.reset()
    all_keys = []
    for i in range(n):
        z = sep(h[i].unsqueeze(0)).squeeze(0)
        memory.store(z, h[i])
        all_keys.append(z)

    # Direct retrieval (no corruption)
    direct_h = []
    for i in range(n):
        h_d = memory.retrieve(all_keys[i].to(device).unsqueeze(0)).squeeze(0)
        direct_h.append(h_d)
    direct_h = torch.stack(direct_h)

    # Corrupted key retrieval (50% active units zeroed)
    retrieved_h = []
    for i in range(n):
        z_q = corrupt_sparse_key(all_keys[i].to(device), level=0.5)
        h_r = memory.retrieve(z_q.unsqueeze(0)).squeeze(0)
        retrieved_h.append(h_r)
    retrieved_h = torch.stack(retrieved_h)

    # Decode all
    with torch.no_grad():
        recon_orig = model.decode(h)
        recon_direct = model.decode(direct_h)
        recon_retrieved = model.decode(retrieved_h)

    fig, axes = plt.subplots(4, n, figsize=(n * 2, 8))
    titles = ['Original', 'Direct memory\n(no corrupt)',
              'Retrieved\n(50% key corrupt)', 'Decoded\n(retrieved)']
    all_ims = [imgs, recon_direct, recon_retrieved, recon_orig]

    for row, (title, row_ims) in enumerate(zip(titles, all_ims)):
        for col in range(n):
            axes[row, col].imshow(row_ims[col, 0].cpu(), cmap='gray',
                                  vmin=0, vmax=1)
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(title, fontsize=8)

    plt.tight_layout()
    path = OUT / 'memory_retrieval_visual.png'
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Hyperparameters
    FEATURE_DIM = 128
    KEY_DIM = 2000
    SPARSITY = 0.02

    # Initialize components
    model = SparseAutoencoder(feature_dim=FEATURE_DIM).to(device)
    sep = PatternSeparator(FEATURE_DIM, KEY_DIM, SPARSITY)
    memory = KeyValueHopfieldMemory(KEY_DIM, FEATURE_DIM, beta=2.0).to(
        torch.device(device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Phase 1: Train autoencoder ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1: Train sparse autoencoder on shapes")
    print(f"feature_dim={FEATURE_DIM}, sparsity_weight={model.sparsity_weight}\n")

    losses = train_autoencoder(model, n_steps=3000, device=device)

    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses, alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Total Loss')
    plt.title('Autoencoder Training Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'training_loss.png', dpi=150)
    plt.close()

    # Visualize reconstruction
    evaluate_reconstruction(model, device=device, title=' After Training')

    # ── Phase 2: Memory retrieval test ────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Memory retrieval with learned features\n")

    metrics = evaluate_memory_retrieval(model, memory, sep, n_samples=200,
                                        device=device, corruption_level=0.5)
    print(f"  Original recon MSE:        {metrics['orig_mse']:.6f}")
    print(f"  Direct memory (no corrupt): {metrics['direct_mse']:.6f}")
    print(f"  Retrieved (50% corrupt):    {metrics['retrieved_mse']:.6f}")
    print(f"  Feature cosim:              {metrics['feature_cosim']:.4f}")

    # Visualize
    visualize_memory_test(model, memory, sep, n=10, device=device)

    # ── Phase 3: Capacity with learned features ───────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Capacity test with learned features\n")

    for n_stored in [10, 50, 100, 200]:
        metrics = evaluate_memory_retrieval(
            model, memory, sep, n_samples=n_stored, device=device,
            corruption_level=0.5)
        print(f"  stored={n_stored:3d} cosim={metrics['feature_cosim']:.4f} "
              f"recon_mse={metrics['retrieved_mse']:.6f}")

    print(f"\nDone. Results in {OUT.resolve()}")


if __name__ == '__main__':
    main()
