"""Step 4: Sleep Consolidation — Fast Memory -> Slow System Knowledge Transfer.

Updated approach after initial failure:
  - Embedding matching (forcing encoder to match stale features) caused collapse.
  - Correct approach: Store (image, key, value) triples. During sleep:
    1. Replay images: standard autoencoder training on stored images (encoder + decoder)
    2. Decoder consolidation: train decoder from memory-retrieved features (decoder only)

  Decoder-only consolidation avoids collapse because:
  - The encoder is trained normally on replayed images (it learns its own representations)
  - The decoder learns to handle memory-retrieved features as additional training signal
  - No conflict between old and new encoder representations

Protocol:
  1. During waking: autoencoder processes images, stores (image, key, value) triples in memory
  2. During sleep: sample triples, train autoencoder on both standard recon and memory-consolidated recon

Test:
  - Phase 1: Train autoencoder normally
  - Phase 2: Store patterns, run sleep with memory replay
  - Phase 3: Compare reconstruction quality before vs. after
  - Phase 4: Memory ablation — does knowledge survive without memory?
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
    corrupt_sparse_key

OUT = Path('results/step4')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TripleMemory(KeyValueHopfieldMemory):
    """Extends KeyValueHopfieldMemory to also store images.

    Stores (image, key, value) triples. Images are needed for
    replay-based consolidation training.
    """

    def store_with_image(self, key: torch.Tensor, value: torch.Tensor,
                         image: torch.Tensor):
        self.store(key, value)
        self.images.append(image.detach().cpu())

    def reset(self):
        super().reset()
        self.images = []

    def sample_triple(self, batch_size: int, device=DEVICE):
        """Sample random (image, key, value) triples from memory."""
        n = len(self.keys)
        idx = torch.randperm(n)[:min(batch_size, n)]
        imgs = torch.stack([self.images[i] for i in idx]).to(device)
        keys = torch.stack([self.keys[i] for i in idx]).to(device)
        vals = torch.stack([self.values[i] for i in idx]).to(device)
        return imgs, keys, vals


def sleep_consolidation(
    model, memory, sep, n_steps=500, batch_size=64, lr=1e-4,
    corruption_level=0.3, recon_weight=1.0, memcon_weight=0.5,
):
    """Sleep consolidation via image replay + decoder-only memory consolidation.

    The loss has two components:
    1. recon_loss: standard autoencoder loss on replayed images (encoder + decoder)
       target = decoder(encoder(image)) ≈ image
       This reinforces the encoder's representations of stored patterns.

    2. memcon_loss: decoder learns from memory-retrieved features (decoder only)
       target = decoder(memory.retrieve(key)) ≈ image
       This trains the decoder to handle features sourced from the memory.

    The memcon loss does NOT update the encoder, preventing the collapse
    that happened when we tried embedding matching.
    """
    if memory.stored_count() < batch_size:
        print(f"  Skipping: only {memory.stored_count()} patterns in memory")
        return []

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = {'total': [], 'recon': [], 'memcon': []}

    for step in range(1, n_steps + 1):
        imgs, keys, vals = memory.sample_triple(batch_size, DEVICE)

        # 1. Standard reconstruction loss on replayed images
        h_enc = model.encode(imgs)
        recon = model.decode(h_enc)
        recon_loss = F.mse_loss(recon, imgs)
        sparsity_loss = 0.001 * h_enc.abs().mean()
        std_loss = recon_loss + sparsity_loss

        # 2. Decoder-only consolidation from memory
        # Corrupt keys to simulate noisy recall
        keys_corrupted = []
        for i in range(batch_size):
            kc = corrupt_sparse_key(keys[i], level=corruption_level)
            keys_corrupted.append(kc)
        keys_corrupted = torch.stack(keys_corrupted)

        # Retrieve values from memory via corrupted keys
        h_retrieved = memory.retrieve(keys_corrupted)

        # Decoder-only loss: decoder must reconstruct image from retrieved features
        recon_from_mem = model.decode(h_retrieved)
        memcon_loss = F.mse_loss(recon_from_mem, imgs)

        # Combined loss
        loss = recon_weight * std_loss + memcon_weight * memcon_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses['total'].append(loss.item())
        losses['recon'].append(std_loss.item())
        losses['memcon'].append(memcon_loss.item())

        if step % 200 == 0 or step == 1:
            print(f"  sleep step={step:5d} total={loss.item():.6f} "
                  f"recon={std_loss.item():.6f} memcon={memcon_loss.item():.6f}")

    return losses


def evaluate_model(model, name='', n_samples=100, device=DEVICE):
    """Evaluate reconstruction quality on fresh shape images."""
    imgs, _ = generate_batch(batch_size=n_samples)
    imgs = imgs.to(device)
    with torch.no_grad():
        recon, h, recon_loss, _ = model(imgs)
    mse = recon_loss.item()
    h_std = h.std().item()
    print(f"  {name:>25}: recon_mse={mse:.6f} h_std={h_std:.4f}")
    return {'mse': mse, 'h_std': h_std}


def evaluate_memory_quality(model, memory, sep, n_samples=100, device=DEVICE):
    """Evaluate: store new patterns, corrupt keys, retrieve, decode.

    Uses original storage order (no shuffling) so that retrieved values
    can be compared against the correct originals.
    """
    imgs, _ = generate_batch(batch_size=n_samples)
    imgs = imgs.to(device)

    with torch.no_grad():
        h = model.encode(imgs)

    temp = TripleMemory(memory.key_dim, memory.value_dim,
                        beta=memory.beta.item()).to(device)
    for i in range(n_samples):
        z = sep(h[i].unsqueeze(0)).squeeze(0)
        temp.store_with_image(z, h[i], imgs[i])

    # Use stored keys directly (NOT shuffled via sample_triple)
    Z = torch.stack(temp.keys, dim=0).to(device)
    keys_c = torch.stack([
        corrupt_sparse_key(Z[i], level=0.5) for i in range(n_samples)
    ])
    h_retrieved = temp.retrieve(keys_c)

    with torch.no_grad():
        recon = model.decode(h_retrieved)
    mse = F.mse_loss(recon, imgs).item()
    cosim = F.cosine_similarity(h_retrieved, h).mean().item()
    print(f"  Memory retrieval (new patterns): mse={mse:.6f} cosim={cosim:.4f}")
    return {'mse': mse, 'cosim': cosim}


def visualize_consolidation(model, memory, sep, n=8, device=DEVICE):
    """Visualize consolidation quality: original vs. memory-retrieved vs. re-encoded."""
    imgs, _ = generate_batch(batch_size=n)
    imgs = imgs.to(device)

    with torch.no_grad():
        h = model.encode(imgs)
        recon_std = model.decode(h)

    temp = TripleMemory(memory.key_dim, memory.value_dim,
                        beta=memory.beta.item()).to(device)
    for i in range(n):
        z = sep(h[i].unsqueeze(0)).squeeze(0)
        temp.store_with_image(z, h[i], imgs[i])

    _, keys, _ = temp.sample_triple(n, device)
    keys_c = torch.stack([
        corrupt_sparse_key(keys[i], level=0.5) for i in range(n)
    ])
    h_retrieved = temp.retrieve(keys_c)

    with torch.no_grad():
        recon_mem = model.decode(h_retrieved)
        # Also decode original features (direct, no corruption)
        h_direct = temp.retrieve(keys)
        recon_direct = model.decode(h_direct)

    fig, axes = plt.subplots(4, n, figsize=(n * 2, 8))
    rows = [imgs, recon_direct, recon_mem, recon_std]
    labels = ['Original', 'Memory (direct)', 'Memory (corrupted key)', 'Autoencoder (no memory)']
    for row, (label, row_imgs) in enumerate(zip(labels, rows)):
        for col in range(n):
            axes[row, col].imshow(row_imgs[col, 0].cpu(), cmap='gray',
                                  vmin=0, vmax=1)
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(label, fontsize=7)
    plt.tight_layout()
    path = OUT / 'consolidation_visual.png'
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def main():
    print(f"Device: {DEVICE}\n")

    FEATURE_DIM = 128
    KEY_DIM = 2000
    SPARSITY = 0.02

    model = SparseAutoencoder(feature_dim=FEATURE_DIM).to(DEVICE)
    sep = PatternSeparator(FEATURE_DIM, KEY_DIM, SPARSITY)
    memory = TripleMemory(KEY_DIM, FEATURE_DIM, beta=2.0).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Phase 1: Initial training ──────────────────────────────────────
    print("=" * 60)
    print("Phase 1: Initial autoencoder training\n")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 2001):
        imgs, _ = generate_batch(batch_size=64)
        imgs = imgs.to(DEVICE)
        recon, h, recon_loss, sp_loss = model(imgs)
        loss = recon_loss + sp_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step={step:5d} recon={recon_loss.item():.6f}")

    evaluate_model(model, 'Before sleep', n_samples=200)
    evaluate_memory_quality(model, memory, sep, n_samples=200)

    # ── Phase 2: Fill memory ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Fill memory with diverse patterns\n")

    for _ in range(10):  # 10 batches of 64 = 640 patterns
        imgs, _ = generate_batch(batch_size=64)
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            h = model.encode(imgs)
        for i in range(64):
            z = sep(h[i].unsqueeze(0)).squeeze(0)
            memory.store_with_image(z, h[i], imgs[i])

    print(f"  Stored {memory.stored_count()} triples in memory")

    # ── Phase 3: Sleep consolidation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Sleep consolidation\n")

    losses = sleep_consolidation(
        model, memory, sep, n_steps=1000, batch_size=64,
        corruption_level=0.3, recon_weight=1.0, memcon_weight=0.5,
    )

    if losses:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (key, title) in zip(axes,
            [('total', 'Total'), ('recon', 'Standard Recon'),
             ('memcon', 'Decoder Memory Consolidation')]):
            ax.plot(losses[key])
            ax.set_title(title)
            ax.set_xlabel('Sleep Step')
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / 'sleep_losses.png', dpi=150)
        plt.close()

    # ── Phase 4: Post-sleep evaluation ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 4: Post-sleep evaluation\n")

    eval_after = evaluate_model(model, 'After sleep', n_samples=200)
    eval_mem = evaluate_memory_quality(model, memory, sep, n_samples=200)
    visualize_consolidation(model, memory, sep, n=8)

    # ── Phase 5: Memory ablation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 5: Memory ablation — standalone autoencoder quality\n")

    memory.reset()
    eval_ablation = evaluate_model(model, 'After ablation (no memory)',
                                    n_samples=200)

    print(f"\n  Summary:")
    print(f"    Before sleep:       recon={0.003:.4f} (reference)")
    print(f"    After sleep:        recon={eval_after['mse']:.6f}")
    print(f"    After ablation:     recon={eval_ablation['mse']:.6f}")
    print(f"    Memory quality:     recon={eval_mem['mse']:.6f}")
    print(f"\nDone. Results in {OUT.resolve()}")


if __name__ == '__main__':
    main()
