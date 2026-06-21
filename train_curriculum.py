"""Full curriculum training: Phase 0→3 with allocentric memory + TD adaptation.
Trains the model across increasing difficulty, then tests generalization.
"""

import time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from collections import deque

from hopfield_memory import PatternSeparator
from env_nav import NavArena
from step9_embodied_learning import (
    ImportanceWeightedMemory, AugmentedPolicyHead, SlowValueNet,
    sleep_consolidation, compute_gae, vicreg_loss, DEVICE, OUT as BASE_OUT
)

OUT = Path('results/curriculum')
OUT.mkdir(parents=True, exist_ok=True)

STATE_DIM = 12
HIDDEN = 64
MEM_SIZE = 1000


def make_env():
    return NavArena(render_mode='rgb_array')


def run_demo(env, memory, dg_pos):
    """Phase 0: Demo from fixed start (0,0) → (3,3). Seeds allocentric memory."""
    env.set_curriculum(0)
    obs, _ = env.reset(seed=42)
    goal = obs['state'][2:4]
    pos = obs['state'][:2]
    data = []
    for _ in range(500):
        delta = goal - pos
        dist = np.linalg.norm(delta)
        if dist > 1.0:
            action = np.clip(delta / dist, -1, 1).astype(np.float32)
        elif dist > 0.3:
            action = np.clip(delta * 0.5, -0.3, 0.3).astype(np.float32)
        else:
            action = np.zeros(2, dtype=np.float32)
        obs2, reward, term, trunc, _ = env.step(action)
        data.append({'state': obs['state'].copy(), 'reward': reward})
        obs = obs2
        pos = obs['state'][:2]
        if term:
            break
    n = len(data)
    qs = [0.0] * n
    cum = 0.0
    for i in range(n - 1, -1, -1):
        cum = data[i]['reward'] + 0.99 * cum
        qs[i] = cum
    for d, q in zip(data, qs):
        s = d['state']
        z = dg_pos(torch.tensor(s[:2], dtype=torch.float32).unsqueeze(0)).squeeze(0)
        memory.keys.append(z.cpu())
        memory.states.append(torch.tensor(s, dtype=torch.float32))
        memory.q_values.append(q)
        memory.importance.append(1.0)
    return n, qs[0]


def train_phase(env, policy, memory, dg_pos, slow_net, slow_opt,
                phase_id, n_steps, policy_opt, success_threshold=3):
    """Train one curriculum phase. Returns metrics dict."""
    env.set_curriculum(phase_id)
    obs, _ = env.reset()
    state_vec = obs['state']

    rollout = {k: [] for k in [
        'state', 'true_gd', 'action', 'log_prob', 'value', 'reward', 'done',
        'state_key', 'ext_rew', 'ca1_nov', 'q_target',
    ]}

    consecutive_success = 0
    total_success = 0
    total_episodes = 0
    step = 0
    ca1_history = deque(maxlen=200)

    while step < n_steps:
        with torch.no_grad():
            st = torch.from_numpy(state_vec).float().to(DEVICE).unsqueeze(0)
            pos, goal = st[:, :2], st[:, 2:4]
            true_gd = (goal - pos) / ((goal - pos).norm(dim=-1, keepdim=True) + 1e-8)

            z = dg_pos(pos)
            state_star, q_star, rbf_sim = memory.retrieve(z)
            ca1_nov = (pos - state_star[:, :2]).norm().item()
            ca1_history.append(ca1_nov)

            pi_in = torch.cat([st.squeeze(0), true_gd.squeeze(0)], dim=-1).unsqueeze(0)
            mean, std, value = policy(pi_in)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            blend = rbf_sim if rbf_sim > 0.05 else 0.0
            q_target = blend * q_star.item() + (1 - blend) * value.item()

        obs2, ext_rew, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
        done = term or trunc
        state_vec2 = obs2['state']

        # TD update on memory
        with torch.no_grad():
            nxt = torch.from_numpy(state_vec2).float().to(DEVICE).unsqueeze(0)
            ngd = (nxt[:, 2:4] - nxt[:, :2]) / ((nxt[:, 2:4] - nxt[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)
            _, _, nv = policy(torch.cat([nxt.squeeze(0), ngd.squeeze(0)], dim=-1).unsqueeze(0))
        memory.td_update(ext_rew, nv.item())

        # Intrinsic: novelty-annealed
        alpha = max(0.05, 0.3 * (1 - step / n_steps))
        total_reward = ext_rew + alpha * ca1_nov

        rollout['state'].append(st.squeeze(0))
        rollout['true_gd'].append(true_gd.squeeze(0))
        rollout['action'].append(action.squeeze(0))
        rollout['log_prob'].append(log_prob)
        rollout['value'].append(value)
        rollout['reward'].append(total_reward)
        rollout['done'].append(1.0 if done else 0.0)
        rollout['state_key'].append(z.squeeze(0))
        rollout['ext_rew'].append(ext_rew)
        rollout['ca1_nov'].append(ca1_nov)
        rollout['q_target'].append(q_target)

        state_vec = state_vec2
        step += 1

        if done:
            total_episodes += 1
            if term:
                consecutive_success += 1
                total_success += 1
                if consecutive_success >= success_threshold:
                    break
            else:
                consecutive_success = 0
            obs, _ = env.reset()
            state_vec = obs['state']

    # PPO update
    if len(rollout['state']) >= 64:
        _do_ppo_update(policy, policy_opt, rollout)

    # Store in memory (allocentric: position → Q_value)
    for t in range(len(rollout['state'])):
        ca1_pct = np.percentile(list(ca1_history) + [0], 60)
        if rollout['ca1_nov'][t] > ca1_pct:
            memory.store(rollout['state_key'][t], rollout['state'][t], rollout['q_target'][t])

    # Sleep
    if len(memory) >= 64:
        sleep_consolidation(memory, slow_net, slow_opt, dg_pos,
                          type('cfg', (), {'minibatch_size': 64})())

    return {
        'steps': step, 'ext': np.mean(rollout['ext_rew']),
        'q_target': np.mean(rollout['q_target']),
        'goal_rate': total_success / max(total_episodes, 1),
        'success': total_success, 'episodes': total_episodes,
        'mem': len(memory),
    }


def _do_ppo_update(policy, opt, rollout):
    """Standard PPO update on rollout data."""
    sb = torch.stack(rollout['state'])
    gb = torch.stack(rollout['true_gd'])
    ab = torch.stack(rollout['action'])
    ob = torch.stack(rollout['log_prob'])
    rb = torch.tensor(rollout['reward'], device=DEVICE, dtype=torch.float32)
    db = torch.tensor(rollout['done'], device=DEVICE, dtype=torch.float32)
    vb = torch.stack(rollout['value'])

    n = sb.size(0)
    if n < 16:
        return

    # GAE
    with torch.no_grad():
        last_s = sb[-1:].clone()
        last_g = gb[-1:].clone()
        _, _, nv = policy(torch.cat([last_s, last_g], dim=-1))
    all_v = torch.cat([vb.view(-1), nv.view(-1)])
    adv, ret = compute_gae(rb, all_v, db, 0.99, 0.95)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(4):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            pi = torch.cat([sb[idx], gb[idx]], dim=-1)
            m, s, vm = policy(pi)
            d = Normal(m, s)
            lp = d.log_prob(ab[idx]).sum(-1)
            ent = d.entropy().sum(-1).mean()

            ratio = (lp - ob[idx]).exp()
            ca = torch.clamp(ratio, 0.8, 1.2) * adv[idx]
            pl = -(torch.min(ratio * adv[idx], ca)).mean()
            vl = F.mse_loss(vm, ret[idx])
            loss = pl + 0.5 * vl - 0.01 * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()


def test_phase(env, policy, memory, dg_pos, phase_id, n_steps=500, label=""):
    """Evaluate on a phase without training. Returns goal rate."""
    env.set_curriculum(phase_id)
    obs, _ = env.reset()
    state_vec = obs['state']
    goals = 0
    episodes = 0

    for _ in range(n_steps):
        with torch.no_grad():
            st = torch.from_numpy(state_vec).float().to(DEVICE).unsqueeze(0)
            pos, goal = st[:, :2], st[:, 2:4]
            gd = (goal - pos) / ((goal - pos).norm(dim=-1, keepdim=True) + 1e-8)
            z = dg_pos(pos)
            _, q_star, rbf = memory.retrieve(z)
            blend = rbf if rbf > 0.05 else 0.0
            pi = torch.cat([st.squeeze(0), gd.squeeze(0)], dim=-1).unsqueeze(0)
            m, s, _ = policy(pi)
            action = Normal(m, s).sample()

        obs2, _, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
        state_vec = obs2['state']
        if term:
            goals += 1
        if term or trunc:
            episodes += 1
            obs, _ = env.reset()
            state_vec = obs['state']

    return goals / max(episodes, 1)


def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("FULL CURRICULUM TRAINING: Phase 0 → 1 → 2 → 3")
    print("=" * 60)

    # ── Components ──────────────────────────────────────────────
    dg_pos = PatternSeparator(2, 2000, 0.02)
    memory = ImportanceWeightedMemory(key_dim=2000, state_dim=STATE_DIM, beta=2.0, max_size=MEM_SIZE)
    policy = AugmentedPolicyHead(input_dim=STATE_DIM + 2, hidden_dim=HIDDEN, action_dim=2).to(DEVICE)
    slow_net = SlowValueNet(state_dim=STATE_DIM).to(DEVICE)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    slow_opt = torch.optim.Adam(slow_net.parameters(), lr=1e-3)

    env = make_env()
    all_metrics = []

    # ══════════════════════════════════════════════════════════════
    # PHASE 0: Demo + Fixed Start Training
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 0: Demo + Fixed Start → Fixed Goal")
    print("=" * 60)
    n, q0 = run_demo(env, memory, dg_pos)
    print(f"  Demo: {n} steps, Q={q0:.1f}, mem={len(memory)}")

    m0 = train_phase(env, policy, memory, dg_pos, slow_net, slow_opt, 0, 2000, policy_opt)
    print(f"  Train: ext={m0['ext']:.2f}, Qtarget={m0['q_target']:.1f}, "
          f"goal_rate={m0['goal_rate']:.0%} ({m0['success']}/{m0['episodes']})")
    all_metrics.append(m0)

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Random Start, Fixed Goal
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("PHASE 1: Random Start → Fixed Goal (learn motor generalization)")
    print(f"{'=' * 60}")
    m1 = train_phase(env, policy, memory, dg_pos, slow_net, slow_opt, 1, 5000, policy_opt)
    print(f"  Train: ext={m1['ext']:.2f}, Qtarget={m1['q_target']:.1f}, "
          f"goal_rate={m1['goal_rate']:.0%} ({m1['success']}/{m1['episodes']})")
    all_metrics.append(m1)

    # Test Phase 1
    t1 = test_phase(env, policy, memory, dg_pos, 1, 1000, "Phase1 test")
    print(f"  TEST random start: goal_rate={t1:.0%}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Random Start, Random Goal
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("PHASE 2: Random Start → Random Goal (memory TD adaptation)")
    print(f"{'=' * 60}")
    m2 = train_phase(env, policy, memory, dg_pos, slow_net, slow_opt, 2, 8000, policy_opt)
    print(f"  Train: ext={m2['ext']:.2f}, Qtarget={m2['q_target']:.1f}, "
          f"goal_rate={m2['goal_rate']:.0%} ({m2['success']}/{m2['episodes']})")
    all_metrics.append(m2)

    # Test Phase 2
    t2 = test_phase(env, policy, memory, dg_pos, 2, 1000, "Phase2 test")
    print(f"  TEST random start+goal: goal_rate={t2:.0%}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Random Start, Random Goal, Random Maze
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("PHASE 3: Random Start → Random Goal → Random Maze Walls")
    print(f"{'=' * 60}")
    m3 = train_phase(env, policy, memory, dg_pos, slow_net, slow_opt, 3, 10000, policy_opt)
    print(f"  Train: ext={m3['ext']:.2f}, Qtarget={m3['q_target']:.1f}, "
          f"goal_rate={m3['goal_rate']:.0%} ({m3['success']}/{m3['episodes']})")
    all_metrics.append(m3)

    # Test Phase 3
    t3 = test_phase(env, policy, memory, dg_pos, 3, 1000, "Phase3 test")
    print(f"  TEST random maze: goal_rate={t3:.0%}")

    # ── Full evaluation across all phases ──────────────────────
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION")
    print(f"{'=' * 60}")
    print(f"{'Phase':<8} {'TrainExt':>8} {'Qtarget':>8} {'GoalRate':>10} {'Mem':>6}")
    print(f"{'-'*40}")
    phase_names = ["Fixed", "RandomStart", "RandomGoal", "RandomMaze"]
    for i, m in enumerate(all_metrics):
        print(f"{phase_names[i]:<8} {m['ext']:>8.2f} {m['q_target']:>8.1f} {m['goal_rate']:>9.0%} {m['mem']:>6}")

    # Final test: all 4 phases
    print(f"\n  Final Tests (1000 steps each):")
    for pid, name in [(0, "Fixed start/goal"), (1, "Random start"), (2, "Random goal"), (3, "Random maze")]:
        rate = test_phase(env, policy, memory, dg_pos, pid, 1000)
        print(f"    {name:<25}: goal_rate={rate:.0%}")

    # ── Save ──────────────────────────────────────────────────
    torch.save(policy.state_dict(), OUT / 'policy.pt')
    torch.save(slow_net.state_dict(), OUT / 'slow_net.pt')
    print(f"\n  Saved to {OUT}")
    env.close()
    print("\nDone.")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    main()
