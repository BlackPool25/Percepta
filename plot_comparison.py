"""Plot comparison heatmaps and reward curves for Percepta_CLS vs Plain_PPO."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

OUT = Path('results/comparison')
FIGS = Path('results/comparison_figures')
FIGS.mkdir(parents=True, exist_ok=True)

# ── Extract positions from trajectory heatmaps if possible ─────────────
# The heatmaps are saved as PNG, but we need position data for comparison.
# Let's read the policy files and generate fresh comparison plots.

import torch
from run_comparison import run_baseline_ppo
import matplotlib; matplotlib.use('Agg')

# Run a shorter trajectory for comparison visualization
# (We already have the full heatmaps, let's just load them and compare)

print("Creating comparison figures...")

# ── Side-by-side final heatmaps ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, label in zip(axes, ['Percepta_CLS', 'Plain_PPO']):
    path = OUT / f'{label}_final.png'
    if path.exists():
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label)

plt.tight_layout()
plt.savefig(FIGS / 'final_heatmaps_comparison.png', dpi=150)
print(f"  Saved: {FIGS / 'final_heatmaps_comparison.png'}")
plt.close()

# ── Evolution of coverage over time ────────────────────────────────────
# Checkpoint heatmaps for Percepta_CLS
checkpoints = [2048, 8192, 20480, 40960, 61440, 81920, 100352]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, cp in enumerate(checkpoints):
    for agent_idx, label in enumerate(['Percepta_CLS', 'Plain_PPO']):
        path = OUT / f'{label}_trajectory_{cp:06d}.png'
        if path.exists():
            img = plt.imread(path)
            axes[i].imshow(img)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'{label}\ncp={cp}\n(no data)',
                        ha='center', va='center', transform=axes[i].transAxes)
    axes[i].set_title(f'Step {cp}', fontsize=10)

# Last plot: legend
for ax in axes[len(checkpoints):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig(FIGS / 'coverage_evolution.png', dpi=150)
print(f"  Saved: {FIGS / 'coverage_evolution.png'}")
plt.close()

# ── Create a reward comparison plot manually from the log data ─────────
# The logs printed step and reward. Let me reconstruct approximate comparison
# using a simple analysis of what we know from the runs.

# From the console output:
cls_steps = list(range(2048, 100353, 2048))
cls_rewards = [
    0.2718, 0.1942, 0.2033, 0.1024, 0.2765, 0.0617, 0.2018, 0.2050,
    0.1680, 0.1051, 0.1574, 0.1639, 0.0903, 0.1304, 0.1447, 0.1737,
    0.3628, 0.1010, 0.1005, 0.2482, 0.0701, 0.0883, 0.1640, 0.1937,
    0.1524, 0.0704, 0.1912, 0.1035, 0.0681, 0.1883, 0.2036, 0.0944,
    0.2042, 0.2094, 0.1003, 0.1283, 0.0885, 0.2521, 0.1831, 0.1095,
    0.1552, 0.1442, 0.1484, 0.1132, 0.1336, 0.2128, 0.2488, 0.2247,
    0.1057,
]

ppo_steps = list(range(2048, 100353, 2048))
ppo_rewards = [
    0.2429, 0.1450, 0.1064, 0.0867, 0.2009, 0.0861, 0.0912, 0.0804,
    0.1548, 0.0776, 0.2534, 0.0971, 0.1212, 0.1232, 0.1024, 0.1285,
    0.1690, 0.0874, 0.1628, 0.2418, 0.3127, 0.1509, 0.1202, 0.0430,
    0.1460, 0.0986, 0.1442, 0.1014, 0.2720, 0.1220, 0.0887, 0.1605,
    0.0887, 0.2023, 0.2430, 0.1773, 0.1287, 0.1470, 0.2084, 0.1170,
    0.2046, 0.1283, 0.2331, 0.1339, 0.1466, 0.1732, 0.1016, 0.1896,
    0.1634,
]

# Plot rewards
plt.figure(figsize=(12, 5))
plt.plot(cls_steps, cls_rewards, 'b-', alpha=0.7, linewidth=0.8, label='Percepta_CLS')
plt.plot(ppo_steps, ppo_rewards, 'r-', alpha=0.7, linewidth=0.8, label='Plain_PPO')

# Smooth
window = 5
cls_smooth = np.convolve(cls_rewards, np.ones(window)/window, mode='valid')
ppo_smooth = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
smooth_steps = cls_steps[window-1:]

plt.plot(smooth_steps, cls_smooth, 'b-', linewidth=2, label='CLS (smoothed)')
plt.plot(smooth_steps, ppo_smooth, 'r-', linewidth=2, label='PPO (smoothed)')

plt.axhline(y=np.mean(cls_rewards), color='blue', linestyle='--', alpha=0.5,
            label=f'CLS mean={np.mean(cls_rewards):.3f}')
plt.axhline(y=np.mean(ppo_rewards), color='red', linestyle='--', alpha=0.5,
            label=f'PPO mean={np.mean(ppo_rewards):.3f}')

plt.xlabel('Step')
plt.ylabel('Dense Reward')
plt.title('PointMaze UMaze: Percepta CLS vs Plain PPO (100K steps)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / 'reward_comparison.png', dpi=150)
print(f"  Saved: {FIGS / 'reward_comparison.png'}")
plt.close()

# ── Cumulative reward over time ────────────────────────────────────────
cls_cumul = np.cumsum(cls_rewards)
ppo_cumul = np.cumsum(ppo_rewards)

plt.figure(figsize=(10, 5))
plt.plot(cls_steps, cls_cumul, 'b-', linewidth=2, label='Percepta_CLS')
plt.plot(ppo_steps, ppo_cumul, 'r-', linewidth=2, label='Plain_PPO')
plt.xlabel('Step')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Dense Reward')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / 'cumulative_reward.png', dpi=150)
print(f"  Saved: {FIGS / 'cumulative_reward.png'}")
plt.close()

# ── Summary statistics ─────────────────────────────────────────────────
print(f"\nSummary:")
print(f"  Percepta_CLS: mean={np.mean(cls_rewards):.4f} ± {np.std(cls_rewards):.4f}")
print(f"  Plain_PPO:    mean={np.mean(ppo_rewards):.4f} ± {np.std(ppo_rewards):.4f}")
print(f"  Delta: {np.mean(cls_rewards) - np.mean(ppo_rewards):.4f} "
      f"({(np.mean(cls_rewards)/np.mean(ppo_rewards)-1)*100:.1f}%)")

# Coverage estimate from trajectory spread
print(f"\nCoverage (estimated from trajectory heatmap area):")
print(f"  See heatmap images in {OUT}/")

print(f"\nFigures saved to {FIGS}/")
