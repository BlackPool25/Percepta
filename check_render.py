"""Render pipeline verification for MuJoCo PointMaze with pixel observations.

Checks:
  1. Frame consistency (consecutive frames differ during random action)
  2. Color channel order (RGB expected)
  3. Normalization (values in [0,1])
  4. Resolution (64x64 or whatever the encoder expects)
  5. Visual inspection (save sample frames as PNGs)
"""

import numpy as np
from PIL import Image
from pathlib import Path

OUT = Path('results/render_check')
OUT.mkdir(parents=True, exist_ok=True)

# ── Create environment ──────────────────────────────────────────────────
import gymnasium as gym
import gymnasium_robotics

# Use the simplest maze: U-Maze, with render_mode="rgb_array" for pixels
env = gym.make(
    'PointMaze_UMaze-v3',
    render_mode='rgb_array',
    continuing_task=False,
)

obs, info = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Observation keys: {obs.keys() if isinstance(obs, dict) else 'not dict'}")
print(f"Render mode: {env.render_mode}")

# ── Get a sample frame ──────────────────────────────────────────────────
frame = env.render()
print(f"\nRaw frame shape: {frame.shape}")
print(f"Raw frame dtype: {frame.dtype}")
print(f"Raw frame min: {frame.min()}, max: {frame.max()}")
print(f"Raw frame mean: {frame.mean():.2f}")

# ── Check 1: Color channel order ────────────────────────────────────────
# RGB images have roughly equal channel means. BGR would have B channel
# higher if there's blue sky, etc.
if frame.ndim == 3 and frame.shape[2] == 3:
    for i, name in enumerate(['R', 'G', 'B']):
        print(f"  Channel {name}: mean={frame[:,:,i].mean():.2f}")
    # Quick sanity: if red channel is systematically much lower than blue,
    # might be BGR
    r, g, b = frame[:,:,0].mean(), frame[:,:,1].mean(), frame[:,:,2].mean()
    if b > r * 1.5 and g > r * 1.5:
        print("  ⚠ Possible BGR ordering (blue/green >> red)")
    else:
        print("  ✓ Channel means look RGB-like")

# ── Check 2: Consecutive frame consistency ──────────────────────────────
print("\nRolling random actions for 20 steps...")
frames = [frame]
for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    frames.append(frame.copy())
    if terminated or truncated:
        break

# Check that consecutive frames differ
print("Consecutive frame differences (L1 norm):")
for i in range(min(5, len(frames) - 1)):
    diff = np.abs(frames[i+1].astype(float) - frames[i].astype(float)).mean()
    print(f"  frame {i}→{i+1}: L1 diff = {diff:.2f}")

all_same = all(
    np.array_equal(frames[0], f) for f in frames[1:]
)
if all_same:
    print("  ❌ CRITICAL: All frames identical — renderer returning cached frames!")
elif np.mean([np.abs(frames[i+1].astype(float) - frames[i].astype(float)).mean()
              for i in range(min(5, len(frames) - 1))]) < 0.1:
    print("  ⚠ Very small frame differences — check if agent is actually moving")
else:
    print("  ✓ Consecutive frames differ — renderer is live")

# ── Check 3: Normalization ─────────────────────────────────────────────
# Typical MuJoCo render returns uint8 [0, 255] or float [0, 1]
if frame.dtype == np.uint8:
    print(f"\n  Frame dtype is uint8 [0, 255]. Will need normalization.")
    frame_norm = frame.astype(np.float32) / 255.0
    print(f"  After /255: min={frame_norm.min():.3f}, max={frame_norm.max():.3f}")
elif frame.dtype == np.float32 or frame.dtype == np.float64:
    print(f"\n  Frame dtype is float. Values in [0,1]: min={frame.min():.3f}, max={frame.max():.3f}")

# ── Check 4: Resolution ────────────────────────────────────────────────
# The autoencoder expects 64×64. Check what we get.
print(f"\n  Render resolution: {frame.shape[1]}x{frame.shape[0]} (WxH)")
print(f"  Autoencoder expects: 64x64")

# ── Check 5: Visual inspection ─────────────────────────────────────────
print("\nSaving sample frames...")
for i in [0, 5, 10, 19]:
    if i < len(frames):
        img = Image.fromarray(frames[i] if frames[i].dtype == np.uint8
                              else (frames[i] * 255).astype(np.uint8))
        path = OUT / f'frame_{i:02d}.png'
        img.save(path)
        print(f"  Saved: {path}")

# Also save a resized version to show what the encoder sees
if frame.shape[:2] != (64, 64):
    print(f"\n  Rendering at {frame.shape[1]}x{frame.shape[0]}, resizing to 64x64 for encoder:")
    small = np.array(Image.fromarray(
        frames[0] if frames[0].dtype == np.uint8
        else (frames[0] * 255).astype(np.uint8)
    ).resize((64, 64)))
    path = OUT / 'resized_64x64.png'
    Image.fromarray(small).save(path)
    print(f"  Saved: {path}")

env.close()
print("\nDone.")
