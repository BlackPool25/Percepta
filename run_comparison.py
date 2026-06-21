"""Comparison: Percepta (CLS) vs Plain PPO baseline on PointMaze UMaze.

Runs both agents for 100K steps, logging trajectories for heatmap comparison.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

OUT = Path('results/comparison')
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Reuse components from step3_mujoco_ppo ───────────────────────────────
from step3_mujoco_ppo import (MazeAutoencoder, PolicyHead, BoundedMemory,
    RunningNorm, compute_gae, make_env, process_obs, Config, AE_PATH, DEVICE)
from torch.distributions import Normal
import torch.nn.functional as F

N_STEPS = 100000
LOG_INTERVAL = 2048


def plot_heatmap(positions, step, label, save_path):
    """Top-down coverage heatmap of agent positions."""
    positions = np.array(positions)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(positions[:, 0], positions[:, 1], gridsize=20, cmap='Blues',
              mincnt=1, alpha=0.8)
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=80,
               marker='*', label='start', zorder=5)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.set_title(f'{label} — step {step}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def run_baseline_ppo(label, use_cls=True, n_steps=N_STEPS):
    """Run PPO with or without CLS novelty signals."""
    print(f"\n===== {label} =====")

    ae = MazeAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(AE_PATH, map_location=DEVICE, weights_only=True))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    policy = PolicyHead().to(DEVICE)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    reward_norm = RunningNorm()
    memory = BoundedMemory(max_size=200) if use_cls else None

    from hopfield_memory import PatternSeparator
    dg = PatternSeparator(128, 2000, 0.02)

    env = make_env()
    obs, info = env.reset()
    frame_t, pos = process_obs(env, obs, info)

    global_step = 0
    sleep_counter = 0
    trajectory = [pos.copy()]
    fast_history = deque(maxlen=500)
    slow_history = deque(maxlen=500)

    while global_step < n_steps:
        # Wake
        for p in ae.parameters():
            p.requires_grad_(False)

        rollout = {k: [] for k in ['h', 'action', 'log_prob', 'value', 'reward', 'done']}

        for _ in range(1024):
            with torch.no_grad():
                recon, h = ae(frame_t.unsqueeze(0))
                z = dg(h)
                mean, std, value = policy(h)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            obs, reward_ext, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            done = term or trunc
            next_frame_t, pos = process_obs(env, obs, info)

            if use_cls:
                slow_nov = F.mse_loss(recon, frame_t.unsqueeze(0)).item()
                fast_nov = memory.fast_novelty(h.squeeze(0))
                fast_history.append(fast_nov)
                slow_history.append(slow_nov)
                reward_int = 0.5 * fast_nov + 0.5 * slow_nov
                total_reward = reward_ext + 0.01 * reward_int
                if len(fast_history) > 20:
                    ft = np.percentile(fast_history, 70)
                    st = np.percentile(slow_history, 70)
                    if (fast_nov > ft) or (slow_nov > st):
                        memory.store(z.squeeze(0), h.squeeze(0), frame_t.cpu())
            else:
                total_reward = reward_ext  # Pure extrinsic, no novelty

            rollout['h'].append(h.squeeze(0))
            rollout['action'].append(action.squeeze(0))
            rollout['log_prob'].append(log_prob)
            rollout['value'].append(value)
            rollout['reward'].append(total_reward)
            rollout['done'].append(1.0 if done else 0.0)

            frame_t = next_frame_t
            global_step += 1
            trajectory.append(pos.copy())

            if done:
                obs, info = env.reset()
                frame_t, pos = process_obs(env, obs, info)

        # PPO update
        hb = torch.stack(rollout['h'])
        ab = torch.stack(rollout['action'])
        olp = torch.stack(rollout['log_prob'])
        vb = torch.stack(rollout['value'])

        for r in rollout['reward']:
            reward_norm.update(r)
        rn = torch.tensor([reward_norm(r) for r in rollout['reward']],
                          device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            next_h = ae.encode(frame_t.unsqueeze(0))
            _, _, next_v = policy(next_h)
        all_v = torch.cat([vb.view(-1), next_v.view(-1)])
        adv, ret = compute_gae(rn, all_v,
                               torch.tensor(rollout['done'], device=DEVICE))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(4):
            perm = torch.randperm(hb.size(0))
            for i in range(0, hb.size(0), 64):
                idx = perm[i:i+64]
                m, s, vm = policy(hb[idx])
                d = Normal(m, s)
                lp = d.log_prob(ab[idx]).sum(dim=-1)
                ent = d.entropy().sum(dim=-1).mean()
                r = (lp - olp[idx]).exp()
                ca = torch.clamp(r, 0.8, 1.2) * adv[idx]
                pl = -(torch.min(r * adv[idx], ca)).mean()
                vl = F.mse_loss(vm, ret[idx])
                loss = pl + 0.5 * vl - 0.01 * ent
                optim.zero_grad()
                loss.backward()
                optim.step()

        # Sleep
        if use_cls:
            sleep_counter += 1
            if sleep_counter % 4 == 0 and len(memory) >= 32:
                for p in ae.parameters():
                    p.requires_grad_(True)
                ae.train()
                eo = torch.optim.Adam(ae.parameters(), lr=1e-5)
                for _ in range(100):
                    om, _, _ = memory.sample(64, DEVICE)
                    if om is None:
                        break
                    rm, _ = ae(om)
                    eo.zero_grad()
                    F.mse_loss(rm, om).backward()
                    eo.step()
                ae.eval()
                for p in ae.parameters():
                    p.requires_grad_(False)
                memory.recompute_keys(ae, dg)
                print(f"  [{label} sleep complete, mem={len(memory)}]")

        if global_step % LOG_INTERVAL == 0:
            mean_r = np.mean(rollout['reward'])
            print(f"  {label:>10} step={global_step:6d} reward={mean_r:.4f}")

        # Trajectory heatmap at checkpoints
        if global_step % 20000 == 0 or global_step >= n_steps - 1024:
            plot_heatmap(trajectory, global_step, label,
                         OUT / f'{label}_trajectory_{global_step:06d}.png')

    env.close()
    plot_heatmap(trajectory, n_steps, label, OUT / f'{label}_final.png')
    torch.save(policy.state_dict(), OUT / f'{label}_policy.pt')
    print(f"  {label} done. {global_step} steps.")


if __name__ == '__main__':
    run_baseline_ppo('Percepta_CLS', use_cls=True)
    run_baseline_ppo('Plain_PPO', use_cls=False)
    print(f"\nResults in {OUT.resolve()}")
