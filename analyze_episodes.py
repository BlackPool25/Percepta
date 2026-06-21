"""Analyze existing run logs — episode-level metrics from PPO_GRU_baseline."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('results/episode_analysis')
OUT.mkdir(parents=True, exist_ok=True)

# Reconstruct from the console output of PPO_GRU_baseline
# Steps are at 2048-step intervals, each covering ~6.8 episodes (2048/300)
# We have the raw extrinsic reward mean per 2048-step window

# From the log:
steps = list(range(2048, 100001, 2048))  # 49 data points
ext_rewards = [
    0.1234, 0.1304, 0.1254, 0.1765, 0.3151, 0.1748, 0.1770, 0.1609,
    0.0734, 0.0797, 0.1045, 0.2811, 0.1695, 0.2783, 0.1445, 0.2977,
    0.3194, 0.1583, 0.2173, 0.1685, 0.3100, 0.1559, 0.1480, 0.2125,
    0.1516, 0.1825, 0.1880, 0.2865, 0.2065, 0.1253, 0.1399, 0.1279,
    0.1566, 0.3150, 0.2403, 0.2616, 0.2024, 0.2050, 0.2594, 0.0842,
    0.0895, 0.1553, 0.2717, 0.0666, 0.0740, 0.2197, 0.1562, 0.1019,
]

# Percepta_CLS rewards from the earlier run for comparison
# Use same length for comparison
n = min(len(ext_rewards), len(cls_rewards), len(steps))
ext_rewards = ext_rewards[:n]
cls_rewards = cls_rewards[:n]
steps = steps[:n]

for arr in [ext_rewards, cls_rewards]:
    arr.insert(0, arr[0])  # heuristic: first window starts at 0

# Rebuild steps
steps = list(range(2048, 2048 * n + 1, 2048))

# 1. Step-reward mean (original) vs rolling median
window = 10
plt.figure(figsize=(12, 5))
plt.plot(steps, ext_rewards, 'r.-', alpha=0.4, label='Raw (2048-mean)')
plt.plot(steps[window-1:], np.convolve(ext_rewards, np.ones(window)/window, mode='valid'),
         'r-', linewidth=2, label='Smoothed')
plt.axhline(y=np.mean(ext_rewards), color='red', linestyle='--', alpha=0.5,
            label=f'Overall mean: {np.mean(ext_rewards):.4f}')
plt.xlabel('Step')
plt.ylabel('Mean Extrinsic Reward per 2048-step Window')
plt.title('PPO_GRU Baseline: Step-Reward Mean (the wrong metric)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'step_reward_mean.png', dpi=150)
print(f"  Saved: {OUT / 'step_reward_mean.png'}")
plt.close()

# 2. Simulate episode-level metrics since we don't have per-episode data
# Approximate: each 2048-step window has ~6.8 episodes (2048/300)
# If mean reward per step is R, and reward = exp(-distance),
# then mean distance ≈ -ln(mean_reward) (inverse, approximate)
# This is crude but gives us a trend estimate

ext_means = np.array(ext_rewards)
# Convert mean reward → approximate mean distance
# reward = exp(-d), so d = -ln(reward) — but this is biased because
# mean(exp(-d)) ≠ exp(-mean(d)). Still, the TREND is informative.
approx_dist = -np.log(np.clip(ext_means, 0.01, 0.99))

cls_means = np.array(cls_rewards)
cls_approx_dist = -np.log(np.clip(cls_means, 0.01, 0.99))

plt.figure(figsize=(12, 5))
plt.plot(steps, approx_dist, 'r.-', alpha=0.4, label='PPO_GRU (approx)')
plt.plot(steps[window-1:], np.convolve(approx_dist, np.ones(window)/window, mode='valid'),
         'r-', linewidth=2, label='PPO_GRU smoothed')

plt.plot(steps, cls_approx_dist, 'b.-', alpha=0.4, label='CLS (approx)')
plt.plot(steps[window-1:], np.convolve(cls_approx_dist, np.ones(window)/window, mode='valid'),
         'b-', linewidth=2, label='CLS smoothed')

plt.xlabel('Step')
plt.ylabel('Approx. Mean Distance to Goal')
plt.title('Estimated Distance Trend (inverse of exp(d) reward)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'approx_distance_trend.png', dpi=150)
print(f"  Saved: {OUT / 'approx_distance_trend.png'}")
plt.close()

# 3. Running min distance (approximate best performance)
# For each 2048-step window, the minimum distance is approximately -ln(max_reward)
# Since we only have mean, we can estimate: in a window of 6 episodes,
# the best episode has distance lower than mean by some factor
# Rough: best_distance ≈ mean_distance - 1.0 (heuristic)
best_dist = approx_dist - 1.0
cls_best_dist = cls_approx_dist - 1.0

plt.figure(figsize=(12, 5))
plt.plot(steps, best_dist, 'r.-', alpha=0.7, label='PPO_GRU (est. best episode)')
plt.plot(steps, cls_best_dist, 'b.-', alpha=0.7, label='CLS (est. best episode)')
# Smoothed
plt.plot(steps[window-1:], np.convolve(best_dist, np.ones(window)/window, mode='valid'),
         'r-', linewidth=2)
plt.plot(steps[window-1:], np.convolve(cls_best_dist, np.ones(window)/window, mode='valid'),
         'b-', linewidth=2)
plt.axhline(y=0.0, color='green', linestyle=':', alpha=0.5, label='Goal reached (d=0)')
plt.xlabel('Step')
plt.ylabel('Est. Best Distance to Goal')
plt.title('Estimated Best-Episode Performance (lower = better)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'best_episode_distance.png', dpi=150)
print(f"  Saved: {OUT / 'best_episode_distance.png'}")
plt.close()

# 4. Summary statistics
print(f"\nSummary:")
print(f"  PPO_GRU baseline:  {len(ext_rewards)} windows × 2048 steps = {len(ext_rewards)*2048} steps")
print(f"  Mean reward:       {np.mean(ext_rewards):.4f} ± {np.std(ext_rewards):.4f}")
print(f"  First 10 windows:  {np.mean(ext_rewards[:10]):.4f}")
print(f"  Last 10 windows:   {np.mean(ext_rewards[-10:]):.4f}")
print(f"  Trend (last-first): {np.mean(ext_rewards[-10:]) - np.mean(ext_rewards[:10]):.4f}")
print(f"\n  CLS (previous run):")
print(f"  Mean reward:       {np.mean(cls_rewards):.4f} ± {np.std(cls_rewards):.4f}")
print(f"  First 10:          {np.mean(cls_rewards[:10]):.4f}")
print(f"  Last 10:           {np.mean(cls_rewards[-10:]):.4f}")
print(f"  Trend:             {np.mean(cls_rewards[-10:]) - np.mean(cls_rewards[:10]):.4f}")

# 5. Plot the reward distribution to see if it's bimodal (near goal vs far)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(ext_rewards, bins=15, alpha=0.7, color='red')
plt.title('PPO_GRU: Reward Distribution\n(2048-step means)')
plt.xlabel('Mean Reward')
plt.ylabel('Count')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(cls_rewards, bins=15, alpha=0.7, color='blue')
plt.title('Percepta_CLS: Reward Distribution\n(2048-step means)')
plt.xlabel('Mean Reward')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'reward_distribution.png', dpi=150)
print(f"  Saved: {OUT / 'reward_distribution.png'}")
plt.close()

print(f"\nPlots saved to {OUT}/")
