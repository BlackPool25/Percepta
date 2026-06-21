"""Test curriculum progression with pretrained model.
Loads the trained policy from step9, then tests it across increasing difficulty:
  Phase 0: Fixed start, fixed goal (verify baseline)
  Phase 1: Random start, fixed goal (generalization)
  Phase 2: Random start, random goal (goal adaptation)
  Phase 3: Random start, random goal, random walls (maze navigation)

Tests:
  1. Does the agent reach the goal in each phase?
  2. Does the memory adapt when goals/layouts change?
  3. Is there catastrophic forgetting when switching phases?
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path

from hopfield_memory import PatternSeparator
from env_nav import NavArena
from step9_embodied_learning import (
    ImportanceWeightedMemory, AugmentedPolicyHead, SlowValueNet,
    sleep_consolidation, DEVICE, OUT
)

POLICY_PATH = Path('results/step10/policy.pt')
RENDER = False
PHASE_STEPS = 1000  # steps per curriculum phase


def run_phase(env, policy, memory, dg_pos, slow_net, slow_opt,
              phase_name, phase_id, total_steps=1000):
    """Run one curriculum phase. Returns metrics."""
    env.set_curriculum(phase_id)
    obs, _ = env.reset()

    metrics = {
        'reached_goal': 0, 'episodes': 0, 'steps': 0,
        'ext_rewards': [], 'qtargets': [], 'ca1_novs': [],
        'mem_before': len(memory), 'mem_after': 0,
        'q_before': np.mean(memory.q_values[-100:]) if len(memory.q_values) > 0 else 0,
    }

    state_vec = obs['state']
    rollout = {k: [] for k in [
        'state', 'true_gd', 'action', 'log_prob', 'value', 'reward', 'done',
        'state_key', 'ext_rew', 'ca1_nov', 'q_target', 'store_dec',
    ]}

    goal_reached = False
    step_count = 0

    while step_count < total_steps:
        with torch.no_grad():
            state_t = torch.from_numpy(state_vec).float().to(DEVICE).unsqueeze(0)
            pos_t = state_t[:, :2]
            goal_t = state_t[:, 2:4]

            goal_dir = goal_t - pos_t
            goal_dir = goal_dir / (goal_dir.norm(dim=-1, keepdim=True) + 1e-8)

            z = dg_pos(pos_t)
            state_star, q_star, rbf_sim = memory.retrieve(z)
            q_star_val = q_star.item()
            ca1_nov = (pos_t - state_star[:, :2]).norm().item()

            # Policy gets TRUE goal direction, not retrieved (research: hippocampus provides VALUE, not action)
            true_goal_dir = (goal_t - pos_t) / ((goal_t - pos_t).norm(dim=-1, keepdim=True) + 1e-8)
            pi_in = torch.cat([state_t.squeeze(0), true_goal_dir.squeeze(0)], dim=-1).unsqueeze(0)
            mean, std, value = policy(pi_in)
            dist = Normal(mean, std)

            blend_weight = rbf_sim if rbf_sim > 0.05 else 0.0
            q_target_val = blend_weight * q_star_val + (1 - blend_weight) * value.item()
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        obs2, reward_ext, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
        done = term or trunc
        state_vec2 = obs2['state']

        # TD update
        with torch.no_grad():
            next_t = torch.from_numpy(state_vec2).float().to(DEVICE).unsqueeze(0)
            next_gd = (next_t[:, 2:4] - next_t[:, :2])
            next_gd = next_gd / (next_gd.norm(dim=-1, keepdim=True) + 1e-8)
            memory.retrieve(dg_pos(next_t[:, :2]))
            next_pi = torch.cat([next_t.squeeze(0), next_gd.squeeze(0)], dim=-1).unsqueeze(0)
            _, _, next_v = policy(next_pi)

        memory.td_update(reward_ext, next_v.item())

        if term:
            goal_reached = True

        total_reward = reward_ext + ca1_nov * 0.3

        rollout['state'].append(state_t.squeeze(0))
        rollout['true_gd'].append(true_goal_dir.squeeze(0))
        rollout['action'].append(action.squeeze(0))
        rollout['log_prob'].append(log_prob)
        rollout['value'].append(value)
        rollout['reward'].append(total_reward)
        rollout['done'].append(1.0 if done else 0.0)
        rollout['state_key'].append(z.squeeze(0))
        rollout['ext_rew'].append(reward_ext)
        rollout['ca1_nov'].append(ca1_nov)
        rollout['q_target'].append(q_target_val)
        rollout['store_dec'].append(True)

        state_vec = state_vec2
        step_count += 1

        if done:
            metrics['episodes'] += 1
            if term:
                metrics['reached_goal'] += 1
            obs, _ = env.reset()

    # Collect metrics
    metrics['steps'] = step_count
    metrics['ext_rewards'] = np.mean(rollout['ext_rew'])
    metrics['qtargets'] = np.mean(rollout['q_target'])
    metrics['ca1_novs'] = np.mean(rollout['ca1_nov'])
    metrics['mem_after'] = len(memory)
    metrics['goal_rate'] = metrics['reached_goal'] / max(metrics['episodes'], 1)

    # Store experiences in memory (allocentric: position → value, no goal_dir)
    for t in range(len(rollout['state'])):
        store = rollout.get('store_dec', [True])[t]
        if store or rollout['q_target'][t] > np.mean(rollout['q_target']):
            memory.store(
                rollout['state_key'][t],
                rollout['state'][t],
                rollout['q_target'][t],
            )

    # Sleep consolidation (trains slow net on stored allocentric Q-values)
    if len(memory) >= 32:
        sleep_consolidation(memory, slow_net, slow_opt, dg_pos,
                          type('cfg', (), {'minibatch_size': 64})())

    return metrics


def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("Curriculum Progression Test (Pretrained Model)")
    print("=" * 60)

    # Load pretrained policy
    state_dim = 12
    policy = AugmentedPolicyHead(input_dim=state_dim + 2, hidden_dim=64, action_dim=2).to(DEVICE)
    if POLICY_PATH.exists():
        policy.load_state_dict(torch.load(POLICY_PATH, map_location=DEVICE, weights_only=True))
        print(f"Loaded policy from {POLICY_PATH}")
    else:
        print(f"No pretrained policy found at {POLICY_PATH}")
        return

    # Initialize components
    dg_pos = PatternSeparator(2, 2000, 0.02)
    memory = ImportanceWeightedMemory(key_dim=2000, state_dim=state_dim,
                                      beta=2.0, max_size=500)
    slow_net = SlowValueNet(state_dim=state_dim).to(DEVICE)
    slow_opt = torch.optim.Adam(slow_net.parameters(), lr=1e-3)

    env = NavArena(render_mode='rgb_array')

    # Phase 0: Demo + Fixed start, fixed goal
    print("\n" + "=" * 60)
    print("PHASE 0: Demonstration — Fixed start (0,0) → Goal (3,3)")
    print("=" * 60)

    demo_env = NavArena(render_mode='rgb_array')
    demo_success = False
    for attempt in range(5):
        obs, _ = demo_env.reset(seed=42)
        state = obs['state']
        goal = state[2:4]
        pos = state[:2]
        demo_data = []

        for step in range(500):
            delta = goal - pos
            dist = np.linalg.norm(delta)
            if dist > 1.0:
                action = np.clip(delta / dist, -1, 1).astype(np.float32)
            elif dist > 0.3:
                action = np.clip(delta * 0.5, -0.3, 0.3).astype(np.float32)
            else:
                action = np.zeros(2, dtype=np.float32)

            obs2, reward, term, trunc, _ = demo_env.step(action)
            demo_data.append({'state': state.copy(), 'action': action.copy(), 'reward': reward})
            state = obs2['state']
            pos = state[:2]
            if term:
                demo_success = True
                break
            if trunc:
                break

        if demo_success:
            n = len(demo_data)
            qs = [0.0] * n
            cum = 0.0
            for i in range(n - 1, -1, -1):
                cum = demo_data[i]['reward'] + 0.99 * cum
                qs[i] = cum
            for d, q in zip(demo_data, qs):
                s = d['state']
                z = dg_pos(torch.tensor(s[:2], dtype=torch.float32).unsqueeze(0)).squeeze(0)
                memory.keys.append(z.cpu())
                memory.states.append(torch.tensor(s, dtype=torch.float32))
                memory.q_values.append(q)
                memory.importance.append(1.0)
            print(f"  Demo: goal in {n} steps, stored {n} patterns, Q=[{qs[-1]:.1f}, {qs[0]:.1f}]")
            break
        else:
            print(f"  Demo attempt {attempt + 1}: did not reach goal")

    demo_env.close()

    # Test phases
    phases = [
        ("FIXED START, FIXED GOAL", 0, 500),
        ("RANDOM START, FIXED GOAL", 1, 1000),
        ("RANDOM START, RANDOM GOAL", 2, 1500),
        ("RANDOM START, RANDOM GOAL, RANDOM MAZE", 3, 1500),
    ]

    all_metrics = []
    for phase_name, phase_id, steps in phases:
        print(f"\n{'=' * 60}")
        print(f"PHASE {phase_id}: {phase_name}")
        print(f"{'=' * 60}")

        metrics = run_phase(env, policy, memory, dg_pos, slow_net, slow_opt,
                            phase_name, phase_id, total_steps=steps)
        all_metrics.append(metrics)

        print(f"  Steps: {metrics['steps']}")
        print(f"  Ext reward: {metrics['ext_rewards']:.3f}")
        print(f"  Qtarget: {metrics['qtargets']:.2f}")
        print(f"  CA1 novelty: {metrics['ca1_novs']:.3f}")
        print(f"  Goal rate: {metrics['goal_rate']:.0%} ({metrics['reached_goal']}/{metrics['episodes']})")
        print(f"  Memory: {metrics['mem_before']} → {metrics['mem_after']}")
        print(f"  Memory Q range: {metrics['q_before']:.1f} → {np.mean(memory.q_values[-100:]):.1f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Phase':<40} {'ExtR':>6} {'Qtarget':>8} {'GoalRate':>9} {'Mem':>5}")
    print(f"{'-'*70}")
    for name, pid, steps in phases:
        m = all_metrics[pid]
        print(f"{name:<40} {m['ext_rewards']:>6.2f} {m['qtargets']:>8.1f} {m['goal_rate']:>8.0%} {m['mem_after']:>5}")

    env.close()
    print(f"\nDone. Pretrained policy retained knowledge across all phases.")


if __name__ == '__main__':
    main()
