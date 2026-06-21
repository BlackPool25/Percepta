"""Step 2: Hopfield Memory with Controlled-Correlation Features.

Tests memory performance when stored patterns have structured correlations
(as real-world learned embeddings will).

Key question: How does within-cluster similarity degrade retrieval accuracy?

Architecture:
  Synthetic features (controlled correlation structure)
    -> DG PatternSeparator -> sparse key z
      -> Key-Value Hopfield memory stores (z_key, h_value)
        -> Retrieval: h* = softmax(beta * z_q @ Z_keys^T) @ H_values

Memory stores key-value pairs:
  key   = pattern-separated sparse code (retrieval addressing)
  value = original feature vector (downstream use)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from hopfield_memory import PatternSeparator

OUT = Path('results/step2')
OUT.mkdir(parents=True, exist_ok=True)


class KeyValueHopfieldMemory:
    """CA3 analogue: stores (key, value) pairs.

    Keys are sparse codes (DG pattern-separated).
    Values are original feature vectors.

    Retrieval: h* = softmax(beta * z_q @ Z_keys^T) @ H_values

    Separates addressing (via sparse keys) from content (via dense values).
    Mirrors hippocampus: DG->CA3 addressing, CA3->CA1 feature retrieval.
    """

    def __init__(self, key_dim: int, value_dim: int, beta: float = 1.0,
                 name: str = ''):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.name = name
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.reset()

    def store(self, key: torch.Tensor, value: torch.Tensor):
        self.keys.append(key.detach().cpu())
        self.values.append(value.detach().cpu())

    def retrieve(self, z_query: torch.Tensor,
                 return_attention: bool = False) -> torch.Tensor | tuple:
        if not self.keys:
            out = z_query.new_zeros(*z_query.shape[:-1], self.value_dim)
            if return_attention:
                return out, None, None
            return out
        Z = torch.stack(self.keys, dim=0).to(z_query.device)
        V = torch.stack(self.values, dim=0).to(z_query.device)
        logits = self.beta * (z_query @ Z.transpose(-1, -2))
        attn = F.softmax(logits, dim=-1)
        out = attn @ V
        if return_attention:
            return out, attn, logits
        return out

    def reset(self):
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def stored_count(self) -> int:
        return len(self.keys)

    def to(self, device):
        self.beta = nn.Parameter(self.beta.to(device))
        return self


# Need to import nn for the Parameter
import torch.nn as nn


def generate_clustered_features(
    num_clusters: int = 10,
    samples_per_cluster: int = 10,
    feature_dim: int = 128,
    within_cluster_std: float = 0.3,
    between_cluster_scale: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic features with controlled correlation structure.

    Lower within_cluster_std -> tighter clusters -> higher within-cluster correlation.
    Higher between_cluster_scale -> more separated clusters.
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(num_clusters, feature_dim) * between_cluster_scale
    data, labels = [], []
    for c in range(num_clusters):
        for _ in range(samples_per_cluster):
            sample = centers[c] + rng.randn(feature_dim) * within_cluster_std
            data.append(sample)
            labels.append(c)
    data = np.array(data, dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / np.clip(norms, 1e-8, None)
    return data, np.array(labels)


def corrupt_pattern(z: torch.Tensor, level: float) -> torch.Tensor:
    """Zero out `level` fraction of active units."""
    z_c = z.clone()
    active = torch.where(z > 0.5)[0]
    if len(active) == 0:
        return z_c
    n = max(1, int(len(active) * level))
    idx = active[torch.randperm(len(active))[:n]]
    z_c[idx] = 0.0
    return z_c


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    a_n = a / (a.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    return (a_n @ b_n.T).mean().item()


def test_correlation_sweep(
    feature_dim=128, key_dim=2000, sparsity=0.02,
    num_clusters=5, samples_per_cluster=10,
    corruption=0.5, beta=1.0, trials=5, device='cpu',
):
    """Retrieval accuracy vs. within-cluster correlation."""
    cluster_stds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    sep = PatternSeparator(feature_dim, key_dim, sparsity)

    results = {}
    for wcs in cluster_stds:
        features, labels = generate_clustered_features(
            num_clusters=num_clusters, samples_per_cluster=samples_per_cluster,
            feature_dim=feature_dim, within_cluster_std=wcs,
        )

        mem_kv = KeyValueHopfieldMemory(key_dim, feature_dim, beta,
                                        name='key-value').to(torch.device(device))
        mem_flat = KeyValueHopfieldMemory(key_dim, feature_dim, beta,
                                          name='flat').to(torch.device(device))

        all_keys, all_vals = [], []
        for i in range(len(features)):
            h = torch.from_numpy(features[i])
            z = sep(h.unsqueeze(0)).squeeze(0)
            mem_kv.store(z, h)
            mem_flat.store(h, h)  # baseline: no pattern separation
            all_keys.append(z)
            all_vals.append(h)

        sims_kv, sims_flat = [], []
        for _ in range(trials):
            n_test = min(50, len(features))
            test_idx = torch.randperm(len(features))[:n_test]
            for idx in test_idx:
                h_orig = all_vals[idx].to(device)
                z_q = corrupt_pattern(all_keys[idx].to(device), corruption)

                h_kv = mem_kv.retrieve(z_q.unsqueeze(0)).squeeze(0)
                h_flat = mem_flat.retrieve(h_orig.unsqueeze(0)).squeeze(0)

                sims_kv.append(cosine_similarity(h_orig.unsqueeze(0),
                                                  h_kv.unsqueeze(0)))
                sims_flat.append(cosine_similarity(h_orig.unsqueeze(0),
                                                    h_flat.unsqueeze(0)))

        results[wcs] = {
            'key_value_cosim': np.mean(sims_kv),
            'flat_cosim': np.mean(sims_flat),
            'key_value_std': np.std(sims_kv),
            'flat_std': np.std(sims_flat),
        }
        print(f"  std={wcs:.3f}: key-value={np.mean(sims_kv):.4f} "
              f"flat={np.mean(sims_flat):.4f}")

    return results


def test_cluster_count_sweep(
    feature_dim=128, key_dim=2000, sparsity=0.02,
    within_cluster_std=0.3, samples_per_cluster=10,
    corruption=0.5, beta=1.0, trials=5, device='cpu',
):
    """Retrieval accuracy vs. number of clusters."""
    cluster_counts = [1, 2, 3, 5, 10, 20, 30, 50]
    sep = PatternSeparator(feature_dim, key_dim, sparsity)

    results = {}
    for nc in cluster_counts:
        total = nc * samples_per_cluster
        features, labels = generate_clustered_features(
            num_clusters=nc, samples_per_cluster=samples_per_cluster,
            feature_dim=feature_dim, within_cluster_std=within_cluster_std,
        )

        mem_kv = KeyValueHopfieldMemory(key_dim, feature_dim, beta,
                                        name='key-value').to(torch.device(device))

        all_keys, all_vals = [], []
        for i in range(len(features)):
            h = torch.from_numpy(features[i])
            z = sep(h.unsqueeze(0)).squeeze(0)
            mem_kv.store(z, h)
            all_keys.append(z)
            all_vals.append(h)

        sims = []
        for _ in range(trials):
            n_test = min(50, total)
            test_idx = torch.randperm(total)[:n_test]
            for idx in test_idx:
                h_orig = all_vals[idx].to(device)
                z_q = corrupt_pattern(all_keys[idx].to(device), corruption)
                h_r = mem_kv.retrieve(z_q.unsqueeze(0)).squeeze(0)
                sims.append(cosine_similarity(h_orig.unsqueeze(0),
                                              h_r.unsqueeze(0)))

        results[nc] = {'cosim': np.mean(sims), 'std': np.std(sims)}
        print(f"  clusters={nc:3d} patterns={total:3d}: "
              f"cosim={np.mean(sims):.4f}")

    return results


def plot_correlation(results, save_path):
    stds = sorted(results.keys())
    kv = [results[s]['key_value_cosim'] for s in stds]
    fl = [results[s]['flat_cosim'] for s in stds]
    plt.figure(figsize=(10, 6))
    plt.semilogx(stds, kv, 'bo-', linewidth=2, label='DG keys (pattern separation)')
    plt.semilogx(stds, fl, 'rs--', linewidth=2, label='Flat keys (no separation)')
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95%')
    plt.xlabel('Within-Cluster Std Dev (lower = tighter clusters)')
    plt.ylabel('Cosine Similarity (retrieved vs. original)')
    plt.title('Effect of Within-Cluster Correlation on Memory Retrieval')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_clusters(results, save_path):
    counts = sorted(results.keys())
    sims = [results[c]['cosim'] for c in counts]
    stds = [results[c]['std'] for c in counts]
    plt.figure(figsize=(10, 6))
    plt.semilogx(counts, sims, 'ro-', linewidth=2)
    plt.fill_between(counts,
                     [s - e for s, e in zip(sims, stds)],
                     [s + e for s, e in zip(sims, stds)],
                     alpha=0.2, color='red')
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cosine Similarity (retrieved vs. original)')
    plt.title('Effect of Cluster Count on Memory Retrieval')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Test 1: Correlation sweep
    print("=" * 60)
    print("TEST: Retrieval vs. within-cluster correlation")
    print("5 clusters, 10 samples/cluster, 50% corruption\n")
    r1 = test_correlation_sweep(device=device)
    plot_correlation(r1, OUT / 'correlation_sweep.png')

    # Test 2: Cluster count sweep
    print("\n" + "=" * 60)
    print("TEST: Retrieval vs. number of clusters")
    print("within_std=0.3, 10 samples/cluster, 50% corruption\n")
    r2 = test_cluster_count_sweep(device=device)
    plot_clusters(r2, OUT / 'cluster_count_sweep.png')

    print(f"\nDone. Plots in {OUT.resolve()}")


if __name__ == '__main__':
    main()
