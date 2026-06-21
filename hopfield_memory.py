"""Step 1: Standalone Modern Hopfield Memory Test.

DG analogue: fixed random projection + k-WTA pattern separation
CA3 analogue: pattern buffer + modern Hopfield (attention) retrieval

Tests:
- Capacity: retrieval accuracy vs. number of stored patterns
- Robustness: retrieval accuracy vs. corruption level
- Sparsity: how active unit fraction affects performance

Neuroscience mapping:
  Entorhinal Cortex → DG: random projection (fixed, never learned)
  DG → CA3: mossy fiber connections (k-WTA sparsification)
  CA3: autoassociative memory via recurrent collaterals (pattern buffer + attention)
  CA3 retrieval: pattern completion from partial cue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class PatternSeparator:
    """Dentate Gyrus analogue.

    Fixed random projection + k-WTA sparsity.
    Maps input features to sparse, decorrelated binary codes.

    The projection matrix P is NEVER learned — fixed at init.
    This matches the DG: strong, fixed mossy fiber connections
    that perform pattern separation via extreme sparsity.
    """

    def __init__(self, input_dim: int, hidden_dim: int, sparsity: float = 0.02):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        k = max(1, int(hidden_dim * sparsity))
        self.k = k

        P = torch.randn(input_dim, hidden_dim)
        P = P / (input_dim ** 0.5)
        self.P = nn.Parameter(P, requires_grad=False)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) → z: (..., hidden_dim) sparse binary."""
        projected = x @ self.P.to(x.device)
        _, indices = torch.topk(projected, self.k, dim=-1)
        z = torch.zeros_like(projected)
        z.scatter_(-1, indices, 1.0)
        return z


class ModernHopfieldMemory:
    """CA3 analogue.

    Pattern buffer with modern Hopfield (attention) retrieval.
    Storage: one-shot append. Retrieval: attention over stored patterns.

    z* = softmax(beta * z_query @ Z^T) @ Z

    Properties:
    - One-shot storage
    - Content-addressable retrieval from partial cues
    - High capacity (exponential in pattern dimension)
    - beta is learnable for gradient-based retrieval modulation
    """

    def __init__(self, pattern_dim: int, beta: float = 1.0):
        self.pattern_dim = pattern_dim
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.reset()

    def store(self, z: torch.Tensor):
        """One-shot Hebbian storage. No gradient flows through stored patterns."""
        self.patterns.append(z.detach().cpu())

    def retrieve(self, z_query: torch.Tensor) -> torch.Tensor:
        """Modern Hopfield retrieval.

        z_query: (..., pattern_dim)
        Returns: (..., pattern_dim) reconstructed pattern
        """
        if not self.patterns:
            return z_query

        Z = torch.stack(self.patterns, dim=0).to(z_query.device)
        attn = F.softmax(self.beta * (z_query @ Z.T), dim=-1)
        return attn @ Z

    def retrieve_via_recurrence(self, z_query: torch.Tensor,
                                 num_steps: int = 5) -> torch.Tensor:
        """Recurrent retrieval via alternating attention + sparsify.

        z_{t+1} = sparsify(softmax(beta * z_t @ Z^T) @ Z)

        This approximates CA3 recurrent dynamics more closely.
        """
        if not self.patterns:
            return z_query

        Z = torch.stack(self.patterns, dim=0).to(z_query.device)
        z = z_query
        for _ in range(num_steps):
            attn = F.softmax(self.beta * (z @ Z.T), dim=-1)
            z = attn @ Z
        return z

    def reset(self):
        self.patterns = []

    def stored_count(self) -> int:
        return len(self.patterns)

    def to(self, device):
        self.beta = nn.Parameter(self.beta.to(device))
        return self


def retrieval_metrics(z_true: torch.Tensor, z_pred: torch.Tensor) -> dict:
    """Precision, recall, F1 for binary sparse retrieval."""
    z_pred_bin = (z_pred > 0.5).float()
    tp = ((z_true.bool() & z_pred_bin.bool()).sum(dim=-1)).float()
    fp = (((~z_true.bool()) & z_pred_bin.bool()).sum(dim=-1)).float()
    fn = ((z_true.bool() & (~z_pred_bin.bool())).sum(dim=-1)).float()

    n_orig = z_true.bool().sum(dim=-1).float().clamp(min=1)
    n_pred = z_pred_bin.bool().sum(dim=-1).float().clamp(min=1)

    precision = (tp / n_pred).mean().item()
    recall = (tp / n_orig).mean().item()
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def corrupt_pattern(z: torch.Tensor, level: float) -> torch.Tensor:
    """Zero out `level` fraction of active units."""
    z_c = z.clone()
    active = torch.where(z > 0.5)[0]
    n = max(1, int(len(active) * level))
    idx = active[torch.randperm(len(active))[:n]]
    z_c[idx] = 0.0
    return z_c


def test_capacity(dim=2000, sparsity=0.02, input_dim=128,
                  max_patterns=1000, corruption=0.5, beta=1.0,
                  trials=5, device='cpu'):
    """Retrieval accuracy vs. number of stored patterns."""
    sep = PatternSeparator(input_dim, dim, sparsity)
    mem = ModernHopfieldMemory(dim, beta).to(torch.device(device))

    max_pt = min(max_patterns, 500)
    points = sorted(set(
        list(range(10, min(101, max_pt + 1), 10)) +
        list(range(min(150, max_pt + 1), max_pt + 1, 50))
    ))

    # Pre-generate all patterns
    rng = torch.Generator()
    rng.manual_seed(42)
    all_patterns = []
    for _ in range(max_patterns):
        x = torch.randn(input_dim, generator=rng)
        z = sep(x.unsqueeze(0)).squeeze(0)
        all_patterns.append(z)

    counts, f1s, stds = [], [], []
    for k in points:
        mem.reset()
        for i in range(k):
            mem.store(all_patterns[i])

        trial_f1 = []
        for _ in range(trials):
            test_idx = torch.randperm(k)[:min(50, k)]
            batch_f1 = []
            for idx in test_idx:
                z_orig = all_patterns[idx].to(device)
                z_q = corrupt_pattern(z_orig, corruption)
                z_r = mem.retrieve(z_q.unsqueeze(0)).squeeze(0)
                m = retrieval_metrics(z_orig.unsqueeze(0), z_r.unsqueeze(0))
                batch_f1.append(m['f1'])
            trial_f1.append(np.mean(batch_f1))

        counts.append(k)
        f1s.append(np.mean(trial_f1))
        stds.append(np.std(trial_f1))
        print(f"  k={k:4d}: F1={f1s[-1]:.4f} ± {stds[-1]:.4f}")

    return counts, f1s, stds


def test_robustness(dim=2000, sparsity=0.02, input_dim=128,
                    num_patterns=100, beta=1.0, device='cpu'):
    """Retrieval accuracy vs. corruption level."""
    sep = PatternSeparator(input_dim, dim, sparsity)
    mem = ModernHopfieldMemory(dim, beta).to(torch.device(device))

    rng = torch.Generator()
    rng.manual_seed(42)
    local_patterns = []
    for _ in range(num_patterns):
        x = torch.randn(input_dim, generator=rng)
        z = sep(x.unsqueeze(0)).squeeze(0)
        mem.store(z)
        local_patterns.append(z)

    levels = np.linspace(0.1, 0.95, 10)
    f1s, stds = [], []
    for lv in levels:
        batch_f1 = []
        test_idx = torch.randperm(num_patterns)[:min(50, num_patterns)]
        for idx in test_idx:
            z_orig = local_patterns[idx].to(device)
            z_q = corrupt_pattern(z_orig, lv)
            z_r = mem.retrieve(z_q.unsqueeze(0)).squeeze(0)
            m = retrieval_metrics(z_orig.unsqueeze(0), z_r.unsqueeze(0))
            batch_f1.append(m['f1'])
        f1s.append(np.mean(batch_f1))
        stds.append(np.std(batch_f1))
        print(f"  corruption={lv:.2f}: F1={f1s[-1]:.4f}")

    return levels.tolist(), f1s, stds


def test_sparsity_sweep(dim=2000, input_dim=128, num_patterns=200,
                         corruption=0.5, beta=1.0, device='cpu'):
    """Retrieval accuracy vs. sparsity level."""
    sparsities = [0.005, 0.01, 0.02, 0.05, 0.10]
    results = {}

    for sp in sparsities:
        print(f"\n  sparsity={sp}")
        c, f1, _ = test_capacity(dim=dim, sparsity=sp, input_dim=input_dim,
                                  max_patterns=num_patterns,
                                  corruption=corruption, beta=beta,
                                  trials=3, device=device)
        results[sp] = (c, f1)

    return results


def plot_capacity(counts, f1s, stds, save_path='capacity_curve.png',
                  title='Modern Hopfield Memory: Capacity Curve'):
    plt.figure(figsize=(10, 6))
    plt.plot(counts, f1s, 'b-', linewidth=2, label='Retrieval F1')
    plt.fill_between(counts,
                     [m - s for m, s in zip(f1s, stds)],
                     [m + s for m, s in zip(f1s, stds)],
                     alpha=0.2, color='blue')
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5,
                label='95% threshold')
    plt.axhline(y=0.99, color='green', linestyle=':', alpha=0.5,
                label='99% threshold')
    plt.xlabel('Number of Stored Patterns')
    plt.ylabel('Retrieval F1 Score')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_robustness(levels, f1s, stds, save_path='robustness_curve.png',
                    title='Modern Hopfield Memory: Robustness to Corruption'):
    plt.figure(figsize=(10, 6))
    plt.plot(levels, f1s, 'r-', linewidth=2, marker='o', label='Retrieval F1')
    plt.fill_between(levels,
                     [m - s for m, s in zip(f1s, stds)],
                     [m + s for m, s in zip(f1s, stds)],
                     alpha=0.2, color='red')
    plt.xlabel('Corruption Level (fraction of active units removed)')
    plt.ylabel('Retrieval F1 Score')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_sparsity_sweep(results, save_path='sparsity_sweep.png'):
    plt.figure(figsize=(12, 6))
    for sp, (counts, f1s) in results.items():
        plt.plot(counts, f1s, linewidth=2, label=f'sparsity={sp}')
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Number of Stored Patterns')
    plt.ylabel('Retrieval F1 Score')
    plt.title('Effect of Sparsity on Memory Capacity')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    out = Path('results/step1')
    out.mkdir(parents=True, exist_ok=True)

    # ── Capacity Test ──────────────────────────────────────────────────
    print("=" * 60)
    print("TEST: Capacity — retrieval accuracy vs. stored pattern count")
    print(f"Dim=2000, Sparsity=2%, Corruption=50%\n")
    counts, f1s, stds = test_capacity(
        dim=2000, sparsity=0.02, input_dim=128,
        max_patterns=500, corruption=0.5, beta=1.0, device=device
    )
    plot_capacity(counts, f1s, stds, str(out / 'capacity_curve.png'))

    # ── Robustness Test ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST: Robustness — retrieval accuracy vs. corruption level")
    print(f"Dim=2000, Sparsity=2%, Patterns=100\n")

    levels, r_f1s, r_stds = test_robustness(
        dim=2000, sparsity=0.02, input_dim=128,
        num_patterns=100, beta=1.0, device=device
    )
    plot_robustness(levels, r_f1s, r_stds, str(out / 'robustness_curve.png'))

    # ── Sparsity Sweep ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST: Sparsity sweep — effect of sparsity on capacity\n")
    results = test_sparsity_sweep(
        dim=2000, input_dim=128, num_patterns=200,
        corruption=0.5, beta=1.0, device=device
    )
    plot_sparsity_sweep(results, str(out / 'sparsity_sweep.png'))

    print("\n" + "=" * 60)
    print("All tests complete.")
    print(f"Plots saved to {out.resolve()}")


if __name__ == '__main__':
    main()
