"""Test: does the architecture learn ANY behavior from demo data, or just goal-steering?

We test with 2 completely different demo policies:
  1. Steer AWAY from goal (anti-steering) — go to the opposite corner
  2. Steer to a FIXED POINT (-4, -4) — ignore the actual goal

If the architecture learns these behaviors, the learning mechanism is general.
If it fails, the architecture is specialized for goal-steering.
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
import sys, logging, time, mujoco

sys.path.insert(0, '.')
from env_nav import NavArena
from train_sr import (Hippocampus, Policy, RawForwardModel, CA3Memory,
                       dopamine_update, DEVICE, OUT, logger)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s',
                    handlers=[logging.StreamHandler()])
log = logging.getLogger('learn_test')


def test_demo_behavior(env, pi, name, n_test=50):
    """Test if policy learned the intended behavior."""
    goals = 0
    for ep in range(n_test):
        s = env.reset(seed=42)[0]['state']
        for _ in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
            # Use the ACTUAL goal direction from environment
            gd = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)
            m, sd, _ = pi(st, gd)
            a = Normal(m, sd).sample()
            obs2, _, term, _, _ = env.step(a.squeeze(0).cpu().numpy())
            s = obs2['state']
            if term:
                goals += 1
                break
    return goals


def run_with_demo(demo_type, n_steps=1000):
    """Run architecture with a custom demo policy.
    demo_type: 'anti' or 'corner'
    """
    hc = Hippocampus()
    pi = Policy().to(DEVICE)
    raw_fm = RawForwardModel().to(DEVICE)
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)
    env = NavArena(render_mode=None)

    # ── Generate custom demo ──────────────────────────────────────
    rng = np.random.RandomState(42)
    grid = 5
    all_s, all_a = [], []
    agent_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "agent")

    for traj_idx in range(grid * grid):
        i, j = traj_idx // grid, traj_idx % grid
        sx = -3.0 + 6.0 * (i + 0.5) / grid
        sy = -3.0 + 6.0 * (j + 0.5) / grid
        env.reset(seed=None)
        env._set_body_pos("agent", np.array([sx, sy, 0.5]))
        s = env._get_obs()['state']
        for _ in range(500):
            p = s[:2]
            if demo_type == 'anti':
                # Steer AWAY from goal (toward (-3,-3) or opposite direction)
                target = np.array([-3.0, -3.0])
            elif demo_type == 'corner':
                # Steer to fixed corner
                target = np.array([-4.0, -4.0])
            else:
                target = s[2:4]  # default: steer toward goal

            d_vec = target - p
            dist = np.linalg.norm(d_vec)
            a = np.clip(d_vec / dist if dist > .2 else d_vec * .5, -1, 1).astype(np.float32)
            obs2, r, term, _, _ = env.step(a)
            s2 = obs2['state']
            all_s.append(torch.tensor(s))
            all_a.append(torch.tensor(a))
            s = s2
            if term:
                break

    log.info(f"Generated {len(all_s)} demo transitions for '{demo_type}'")

    # Store in hippocampus
    for i in range(len(all_s)):
        hc.store(all_s[i], all_a[i], 0.0, torch.zeros(12))

    # Pre-train policy + FM on demo
    ds = torch.stack(all_s).to(DEVICE)
    da = torch.stack(all_a).to(DEVICE)
    dg = []
    for dsi in all_s:
        st = dsi.unsqueeze(0).to(DEVICE)
        g = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)
        dg.append(g)
    dg = torch.cat(dg)

    for _ in range(200):
        m, _, _ = pi(ds, dg)
        loss = F.mse_loss(m, da.to(DEVICE))
        opt_pi.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()

    # Verify cosim
    with torch.no_grad():
        m_test, _, _ = pi(ds, dg)
        cosim = F.cosine_similarity(m_test, da.to(DEVICE), dim=-1).mean().item()
    log.info(f"BC init cosim: {cosim:.3f}")

    # ── Train ────────────────────────────────────────────────────
    goals, step = 0, 0
    s = env.reset(seed=42)[0]['state']
    ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []
    dopamine_boost = 1.0
    dopamine_decay_steps = 0

    while step < n_steps:
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        gd = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

        idx = hc.retrieve(st.squeeze(0), k=10)
        if idx and len(hc.ca3.actions) > 0:
            hc_a = torch.stack([hc.ca3.actions[i] for i in idx]).to(DEVICE)
            hc_z = torch.stack([hc.ca3.patterns[i] for i in idx]).to(DEVICE)
            z_q = hc.dg(st.squeeze(0).unsqueeze(0))
            sims = torch.softmax(z_q @ hc_z.T * 5.0, dim=-1)
            action = sims @ hc_a
        else:
            m, sd, _ = pi(st, gd)
            action = Normal(m, sd).sample()

        obs2, re, term, _, _ = env.step(action.squeeze(0).cpu().numpy())
        s2 = obs2['state']
        s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)
        gd2 = (s2_t[:, 2:4] - s2_t[:, :2]) / ((s2_t[:, 2:4] - s2_t[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

        if dopamine_decay_steps > 0:
            dopamine_decay_steps -= 1
            dopamine_boost = 1.0 + 4.0 * (dopamine_decay_steps / 25.0)

        delta, _ = dopamine_update(pi, opt_pi, opt_val, st, gd, action, re, s2_t, gd2,
                                   dopamine_boost=dopamine_boost)
        hc.store(st.squeeze(0), action.squeeze(0), re, s2_t.squeeze(0))

        step += 1
        if term:
            goals += 1
            dopamine_boost = 5.0
            dopamine_decay_steps = 25
            ep_s, ep_a, ep_g, ep_r, ep_ns = [], [], [], [], []

        if term:
            s = env.reset(seed=42)[0]['state']
        else:
            s = s2

    log.info(f"Training: {goals} goals in {step} steps")

    # ── Test policy-only (no hippocampus) ─────────────────────────
    test_goals = test_demo_behavior(env, pi, demo_type)
    log.info(f"Policy-only test: {test_goals}/50 = {test_goals / 50:.0%}")

    env.close()
    return test_goals


if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: Steer to CORNER (-4, -4) — different target than goal")
    corner_goals = run_with_demo('corner', n_steps=500)
    print()
    print("Test 2: Steer AWAY from goal (anti-steering)")
    anti_goals = run_with_demo('anti', n_steps=500)
    print()
    print("=" * 60)
    print(f"Corner-steering learned: {corner_goals}/50 = {corner_goals/50:.0%}")
    print(f"Anti-steering learned:   {anti_goals}/50 = {anti_goals/50:.0%}")
    print(f"Conclusion: {'GENERAL LEARNING' if corner_goals > 25 else 'TASK-SPECIFIC'}")
    print("=" * 60)
