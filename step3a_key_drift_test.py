"""Step 3a: Memory Stability Under Key Drift.

Tests what happens when stored keys drift over time — simulating
encoder representation drift during training.

This was the root cause of the old RSSM failure: the encoder's
features changed, stored patterns became stale, and retrieval
returned noise.

Scenario:
  1. Generate initial features, store (key, value) pairs in memory
  2. Gradually drift ALL features (simulating encoder weight changes)
  3. At each drift step: query with the drifted feature's key
  4. Measure: does memory still retrieve the CORRECT original value?

Key question: How much drift can the DG + Hopfield memory tolerate
before retrieval degrades to chance?
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

OUT = Path('results/step3a')
OUT.mkdir(parents=True, exist_ok=True)


class KeyValueHopfieldMemory:
    """Same as in step2 — stores (key, value) pairs."""

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

    def retrieve(self, z_query: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return z_query.new_zeros(*z_query.shape[:-1], self.value_dim)
        Z = torch.stack(self.keys, dim=0).to(z_query.device)
        V = torch.stack(self.values, dim=0).to(z_query.device)
        attn = F.softmax(self.beta * (z_query @ Z.T), dim=-1)
        return attn @ V

    def reset(self):
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def to(self, device):
        self.beta = nn.Parameter(self.beta.to(device))
        return self


def generate_clustered_features(
    num_clusters, samples_per_cluster, feature_dim=128,
    within_cluster_std=0.3, between_cluster_scale=2.0, seed=42,
):
    """Synthetic features with controlled correlation structure."""
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


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between each pair of vectors in two batches."""
    a_n = a / (a.norm(p=2, dim=1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=1)


def compute_retrieval_quality(memory, query_keys, original_values, device):
    """Measure retrieval quality by cosine similarity to original values."""
    cosims = []
    for i in range(query_keys.size(0)):
        z_q = query_keys[i].unsqueeze(0)
        h_orig = original_values[i].unsqueeze(0).to(device)
        h_ret = memory.retrieve(z_q)
        cosim = F.cosine_similarity(h_ret, h_orig).item()
        cosims.append(cosim)
    mean_cosim = np.mean(cosims)
    frac_above_095 = np.mean([c > 0.95 for c in cosims])
    frac_above_099 = np.mean([c > 0.99 for c in cosims])
    return {
        'mean_cosim': mean_cosim,
        'frac_above_095': frac_above_095,
        'frac_above_099': frac_above_099,
    }


def test_drift_tolerance(
    feature_dim=128, key_dim=2000, sparsity=0.02,
    num_clusters=5, samples_per_cluster=10,
    drift_steps=100, drift_rate=0.01, drift_noise=0.001,
    beta=1.0, device='cpu',
):
    """Test retrieval accuracy as features drift away from stored keys.

    The stored feature vectors remain at their original values.
    The query keys drift each step (simulating encoder drift).
    At each step, we measure: can we retrieve the correct original value?

    Args:
        drift_rate: per-step deterministic drift coefficient
        drift_noise: per-step random walk noise stddev
    """
    sep = PatternSeparator(feature_dim, key_dim, sparsity)
    mem = KeyValueHopfieldMemory(key_dim, feature_dim, beta).to(
        torch.device(device))

    # Generate initial features
    features_raw, labels = generate_clustered_features(
        num_clusters=num_clusters, samples_per_cluster=samples_per_cluster,
        feature_dim=feature_dim,
    )
    n_patterns = len(features_raw)

    # Store (key, value) pairs using INITIAL features
    original_values = []
    for i in range(n_patterns):
        h = torch.from_numpy(features_raw[i]).float()
        z = sep(h.unsqueeze(0)).squeeze(0)
        mem.store(z, h)
        original_values.append(h)

    original_values = torch.stack(original_values)

    # Track drifted features
    drifted = torch.from_numpy(features_raw).float()

    drift_magnitudes = []
    retrieval_cosims = []
    top1_accs = []

    # Before any drift — baseline
    queries = []
    for i in range(n_patterns):
        h = drifted[i]
        z = sep(h.unsqueeze(0)).squeeze(0)
        queries.append(z)
    q = torch.stack(queries).to(device)
    qual = compute_retrieval_quality(mem, q, original_values, device)
    drift_magnitudes.append(0.0)
    retrieval_cosims.append(qual['mean_cosim'])
    top1_accs.append(qual['frac_above_099'])

    # Gradual drift
    for step in range(1, drift_steps + 1):
        # Drift each feature: deterministic decay toward zero + random walk
        noise = torch.randn_like(drifted) * drift_noise
        drifted = drifted * (1.0 - drift_rate) + noise
        # Renormalize
        drifted = F.normalize(drifted, dim=1)

        # Measure total drift magnitude
        drift_mag = (drifted - original_values).norm(dim=1).mean().item()

        # Compute query keys from drifted features
        queries = []
        for i in range(n_patterns):
            h = drifted[i]
            z = sep(h.unsqueeze(0)).squeeze(0)
            queries.append(z)
        q = torch.stack(queries).to(device)

        # Test retrieval against ORIGINAL values
        qual = compute_retrieval_quality(mem, q, original_values, device)
        cosim = qual['mean_cosim']
        frac_099 = qual['frac_above_099']

        drift_magnitudes.append(drift_mag)
        retrieval_cosims.append(cosim)
        top1_accs.append(frac_099)

        if step % 20 == 0 or step == 1:
            print(f"  step={step:3d} drift={drift_mag:.4f} "
                  f"cosim={cosim:.4f} >099={frac_099:.3f}")

    # Also compute: what's the cosine similarity between drifted key and stored key?
    stored_keys = torch.stack(mem.keys, dim=0).to(device)
    original_keys = stored_keys.clone()
    # We already store keys, so compute key drift too
    key_agreements = []
    for i in range(n_patterns):
        z_stored = stored_keys[i]
        z_drifted_q = torch.stack(queries, dim=0).to(device)[i]
        # Fraction of active units that overlap
        overlap = (z_stored.bool() & z_drifted_q.bool()).sum().item()
        n_stored = z_stored.bool().sum().item()
        key_agreements.append(overlap / max(n_stored, 1))
    mean_key_overlap = np.mean(key_agreements)

    return {
        'drift_magnitudes': drift_magnitudes,
        'retrieval_cosims': retrieval_cosims,
        'top1_accs': top1_accs,
        'final_key_overlap': mean_key_overlap,
    }


def test_drift_rate_sweep(
    feature_dim=128, key_dim=2000, sparsity=0.02,
    num_clusters=5, samples_per_cluster=10,
    drift_steps=200, drift_noise=0.001,
    beta=1.0, device='cpu',
):
    """Test drift tolerance across different drift rates."""
    drift_rates = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    results = {}

    for dr in drift_rates:
        print(f"\n  drift_rate={dr}")
        r = test_drift_tolerance(
            feature_dim=feature_dim, key_dim=key_dim, sparsity=sparsity,
            num_clusters=num_clusters, samples_per_cluster=samples_per_cluster,
            drift_steps=drift_steps, drift_rate=dr, drift_noise=drift_noise,
            beta=beta, device=device,
        )
        results[dr] = r

    return results


def plot_drift_decay(results, save_path):
    plt.figure(figsize=(12, 6))
    for dr, r in results.items():
        dm = r['drift_magnitudes']
        cs = r['retrieval_cosims']
        plt.plot(dm, cs, linewidth=2, label=f'drift_rate={dr}')

    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% cosim')
    plt.axhline(y=0.50, color='red', linestyle=':', alpha=0.5, label='50%')
    plt.xlabel('Mean Feature Drift (||h_drifted - h_original||)')
    plt.ylabel('Retrieval Cosine Similarity')
    plt.title('Memory Retrieval vs. Feature Drift')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_drift_over_time(results, save_path):
    plt.figure(figsize=(12, 6))
    for dr, r in results.items():
        steps = list(range(len(r['retrieval_cosims'])))
        plt.plot(steps, r['retrieval_cosims'], linewidth=2,
                 label=f'drift_rate={dr}')
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Drift Step')
    plt.ylabel('Retrieval Cosine Similarity')
    plt.title('Memory Retrieval Decay Over Time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def print_summary(results):
    print("\nSummary:")
    print(f"{'drift_rate':>10} {'final_drift':>12} {'final_cosim':>12} "
          f"{'key_overlap':>12}")
    for dr, r in sorted(results.items()):
        fd = r['drift_magnitudes'][-1]
        fc = r['retrieval_cosims'][-1]
        ko = r.get('final_key_overlap', 0)
        print(f"{dr:>10.4f} {fd:>12.4f} {fc:>12.4f} {ko:>12.3f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ── Test 1: Single drift rate, detailed view ──────────────────────
    print("=" * 60)
    print("TEST: Drift tolerance at rate=0.01")
    print("5 clusters, 10 samples/cluster, 200 steps\n")
    r_single = test_drift_tolerance(
        drift_rate=0.01, drift_steps=200, device=device)

    # ── Test 2: Drift rate sweep ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST: Drift rate sweep")
    print("5 clusters, 10 samples/cluster, 200 steps\n")
    results = test_drift_rate_sweep(
        drift_steps=200, device=device)

    # ── Plot ──────────────────────────────────────────────────────────
    plot_drift_decay(results, OUT / 'drift_decay.png')
    plot_drift_over_time(results, OUT / 'drift_over_time.png')

    print_summary(results)

    print(f"\nDone. Plots in {OUT.resolve()}")


if __name__ == '__main__':
    main()
