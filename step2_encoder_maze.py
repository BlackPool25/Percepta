"""Step 2: Train sparse autoencoder on maze frames with VICReg + systematic coverage.

Fixes:
  - VICReg variance loss: forces each feature dim to carry signal (prevents collapse)
  - VICReg covariance loss: forces dims to be decorrelated (prevents rank-1 collapse)
  - Systematic maze coverage: collect frames at fixed grid positions, not random rollouts
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

OUT = Path('results/step2_maze')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from step3b_sparse_autoencoder import SparseAutoencoder


class MazeAutoencoder(SparseAutoencoder):
    """3-channel autoencoder with BatchNorm to prevent feature collapse."""

    def __init__(self, feature_dim=128, sparsity_weight=0.0, img_size=64):
        super().__init__(feature_dim, sparsity_weight, img_size)
        enc_list = list(self.encoder.children())
        enc_list[0] = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.encoder = nn.Sequential(*enc_list)
        dec_list = list(self.decoder.children())
        dec_list[-2] = nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2, output_padding=1)
        self.decoder = nn.Sequential(*dec_list)
        self.encoder_bn = nn.BatchNorm1d(feature_dim)

    def encode(self, x):
        return self.encoder_bn(self.encoder(x))


def vicreg_loss(h, var_weight=1.0, cov_weight=0.04):
    """VICReg: variance + covariance regularization.

    Variance: max(0, 1 - std(h)) — forces each dim to carry signal.
    Covariance: off-diagonal → 0 — prevents dims from correlating.
    """
    std = torch.sqrt(h.var(dim=0) + 1e-4)
    var_loss = torch.mean(torch.clamp(1 - std, min=0))

    h_c = h - h.mean(dim=0)
    cov = (h_c.T @ h_c) / (h.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / h.size(1)

    return var_weight * var_loss + cov_weight * cov_loss


def resize_frame(frame, target_size=64):
    img = Image.fromarray(frame).resize((target_size, target_size))
    return np.array(img, dtype=np.float32) / 255.0


def collect_systematic_frames(device=DEVICE):
    """Collect maze frames covering the full maze systematically.

    Uses a heuristic to drive the agent to different maze regions:
    sets the goal to a target position and lets the built-in dynamics
    move the agent there, collecting frames along the way.
    
    Also collects from random rollouts for diversity.
    """
    env = gym.make('PointMaze_UMaze-v3', render_mode='rgb_array',
                   continuing_task=False)
    frames = []

    # Phase 1: Systematic coverage via goal-reaching
    print("  Phase 1: Systematic goal-directed coverage...")
    # U-Maze has a U shape. Key positions to cover:
    # bottom-left (start), bottom-right, top-right, top-left
    # The maze is 5x5 grid cells, each cell is 1 unit
    target_positions = [
        np.array([0.5, 0.5]),  # start (bottom-left)
        np.array([3.5, 0.5]),  # bottom-right
        np.array([3.5, 3.5]),  # top-right
        np.array([0.5, 3.5]),  # top-left
        np.array([1.5, 2.0]),  # center of left corridor
        np.array([2.5, 0.5]),  # middle of bottom
        np.array([3.5, 1.5]),  # right side middle
    ]
    for target in target_positions:
        obs, info = env.reset()
        for step in range(200):
            # Move toward target using simple heuristic
            pos = obs['observation'][:2]
            delta = target - pos
            dist = np.linalg.norm(delta)
            if dist < 0.5:
                break
            action = np.clip(delta / max(dist, 0.1) * 0.5, -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)
            if step % 3 == 0:  # Sample every 3rd frame
                raw = env.render()
                frames.append(resize_frame(raw, 64))
            if terminated or truncated:
                break

    # Phase 2: Random exploration for variety
    print(f"  Phase 2: Random exploration (have {len(frames)} frames)...")
    for ep in range(5):
        obs, info = env.reset()
        for step in range(200):
            raw = env.render()
            frames.append(resize_frame(raw, 64))
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if len(frames) >= 3000:
                break
            if terminated or truncated:
                break
        if len(frames) >= 3000:
            break

    env.close()

    # Limit to 3000 frames
    frames = frames[:3000]
    result = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float().to(device)
    print(f"  Total: {len(frames)} frames, shape {result.shape}")
    return result


def main():
    print(f"Device: {DEVICE}\n")

    model = MazeAutoencoder(feature_dim=128, sparsity_weight=0.0).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Collect diverse frames ────────────────────────────────────────
    print("Collecting maze frames with systematic coverage...")
    frames = collect_systematic_frames()
    n_frames = frames.size(0)
    split = int(n_frames * 0.8)
    train_frames = frames[:split]
    test_frames = frames[split:]
    print(f"  Train: {train_frames.size(0)}, Test: {test_frames.size(0)}")

    # ── Train with VICReg ─────────────────────────────────────────────
    print("\nTraining with VICReg loss...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_weight = 1.0
    vicreg_weight = 0.1  # Increased from 0.01 — need strong push against collapse

    for epoch in range(80):
        perm = torch.randperm(train_frames.size(0))
        epoch_recon = 0
        epoch_vicreg = 0

        for i in range(0, train_frames.size(0), 64):
            batch = train_frames[perm[i:i+64]]
            recon, h, recon_loss, _ = model(batch)
            r_loss = recon_loss
            v_loss = vicreg_loss(h, var_weight=1.0, cov_weight=0.04)
            loss = recon_weight * r_loss + vicreg_weight * v_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_recon += r_loss.item()
            epoch_vicreg += v_loss.item()

        if epoch % 10 == 0 or epoch == 0:
            with torch.no_grad():
                r, h_test, rl, _ = model(test_frames)
                test_mse = rl.item()
                v_test = vicreg_loss(h_test).item()
                h_std = h_test.std(dim=0).mean().item()
                h_var = (h_test.std(dim=0) > 0.5).float().mean().item()

                # Check effective rank
                h_np = h_test.cpu().numpy()
                centered = h_np - h_np.mean(axis=0)
                u, s, vt = np.linalg.svd(centered, full_matrices=False)
                var_pc1 = s[0]**2 / (s**2).sum()
                var_pc5 = (s[:5]**2).sum() / (s**2).sum()

            print(f"  ep {epoch:2d}: recon={test_mse:.6f} vicreg={v_test:.6f} "
                  f"h_std={h_std:.4f} >0.5={h_var:.2f} "
                  f"PC1={var_pc1:.0%} PC5={var_pc5:.0%}")

    # ── Final evaluation ──────────────────────────────────────────────
    with torch.no_grad():
        recon, h, recon_loss, _ = model(test_frames)
        h_std = h.std(dim=0).mean().item()
        h_mean = h.mean(dim=0).mean().item()

        h_np = h.cpu().numpy()
        centered = h_np - h_np.mean(axis=0)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        var_pc1 = s[0]**2 / (s**2).sum()
        var_pc5 = (s[:5]**2).sum() / (s**2).sum()

    print(f"\n  Final:")
    print(f"    h_mean={h_mean:.4f} h_std={h_std:.4f}")
    print(f"    PC1={var_pc1:.0%} PC1-5={var_pc5:.0%}")
    print(f"    recon MSE={recon_loss.item():.6f}")

    if h_std > 0.5 and var_pc1 < 0.5:
        print(f"\n  ✓ Pass: diverse features with good decorrelation")
    elif h_std > 0.3:
        print(f"\n  ⚠ Marginal: some signal but may need tuning")
    else:
        print(f"\n  ❌ Failed: features still collapsed")

    # Save
    torch.save(model.state_dict(), OUT / 'autoencoder_maze.pt')
    print(f"  Saved: {OUT / 'autoencoder_maze.pt'}")

    # Sample reconstructions
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\nReconstructions:")
    n_show = 5
    fig, axes = plt.subplots(2, n_show, figsize=(2*n_show, 4))
    for i in range(n_show):
        orig = test_frames[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig)
        axes[0, i].axis('off')
        rec = recon[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Recon')
    plt.tight_layout()
    plt.savefig(OUT / 'maze_reconstructions.png', dpi=150)
    print(f"  Saved: {OUT / 'maze_reconstructions.png'}")
    plt.close()

    print(f"\nDone.")


if __name__ == '__main__':
    main()
