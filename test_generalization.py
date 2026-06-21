"""Generalization test: 4 phases using hippocampal episodic control.
Tests whether the hippocampal memory + policy generalize to:
  Phase 0: Fixed start → Fixed goal (baseline)
  Phase 1: Random start → Fixed goal (generalization to novel positions)
  Phase 2: Random start → Random goal (generalization to novel positions + goals)
  Phase 3: Random start → Random goal + Random maze walls (full generalization)

Each phase runs 50 episodes. Metrics: goals reached, steps to goal.
"""

import torch, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from env_nav import NavArena
from train_sr import Hippocampus, Policy, RawForwardModel, DEVICE, logger
import logging, time

OUT = Path('results/test_gen')
OUT.mkdir(parents=True, exist_ok=True)


def run_phase(env, hc, pi, raw_fm, phase_id, n_episodes=50, name=""):
    """Run episodes with hippocampal action selection."""
    env.set_curriculum(phase_id)
    goals, total_steps, steps_to_goal = 0, 0, []
    t0 = time.time()

    for ep in range(n_episodes):
        s = env.reset(seed=42 if phase_id == 0 else None)[0]['state']
        reached = False
        ep_steps = 0

        for _ in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
            gd = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

            idx = hc.retrieve(st.squeeze(0), k=10)
            if idx and len(hc.ca3.actions) > 0:
                hc_a = torch.stack([hc.ca3.actions[i] for i in idx]).to(DEVICE)
                hc_z = torch.stack([hc.ca3.patterns[i] for i in idx]).to(DEVICE)
                z_q = hc.dg(st.squeeze(0).unsqueeze(0))
                sims = torch.softmax(z_q @ hc_z.T * 5.0, dim=-1)
                a = sims @ hc_a
            else:
                m, sd, _ = pi(st, gd)
                a = Normal(m, sd).sample()

            obs2, _, term, _, _ = env.step(a.squeeze(0).cpu().numpy())
            s = obs2['state']
            ep_steps += 1
            if term:
                goals += 1
                steps_to_goal.append(ep_steps)
                reached = True
                break

        total_steps += ep_steps

    elapsed = time.time() - t0
    avg_steps = sum(steps_to_goal) / len(steps_to_goal) if steps_to_goal else 0
    logger.info(f"[{name}] Phase {phase_id}: {goals}/{n_episodes} = {goals/n_episodes:.0%}, "
                f"avg {avg_steps:.0f} steps, {elapsed:.1f}s")
    return goals, steps_to_goal


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger('gen_test')

    # Load trained models
    SR_OUT = Path('results/sr')
    hc = Hippocampus()
    hc.load_state_dict(torch.load(SR_OUT / 'hc.pt', map_location=DEVICE))
    pi = Policy().to(DEVICE)
    pi.load_state_dict(torch.load(SR_OUT / 'policy.pt', map_location=DEVICE))
    raw_fm = RawForwardModel().to(DEVICE)
    raw_fm.load_state_dict(torch.load(SR_OUT / 'raw_fm.pt', map_location=DEVICE))

    logger.info(f"Loaded hippocampus with {len(hc)} patterns")

    env = NavArena(render_mode=None)
    results = {}

    # Phase 0: Fixed start → Fixed goal (baseline)
    g, s = run_phase(env, hc, pi, raw_fm, 0, 50, "Baseline")
    results['phase0'] = {'goals': g, 'steps': s}

    # Phase 1: Random start → Fixed goal
    g, s = run_phase(env, hc, pi, raw_fm, 1, 50, "RandomStart")
    results['phase1'] = {'goals': g, 'steps': s}

    # Phase 2: Random start → Random goal
    g, s = run_phase(env, hc, pi, raw_fm, 2, 50, "RandomGoal")
    results['phase2'] = {'goals': g, 'steps': s}

    # Phase 3: Random start → Random goal + Random maze
    g, s = run_phase(env, hc, pi, raw_fm, 3, 50, "RandomMaze")
    results['phase3'] = {'goals': g, 'steps': s}

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Phase':20s} {'Success':>10s} {'Avg Steps':>10s}")
    print("-" * 40)
    for k, v in results.items():
        avg_s = sum(v['steps']) / len(v['steps']) if v['steps'] else 0
        print(f"{k:20s} {v['goals']}/50 = {v['goals']/50:.0%} -> {avg_s:5.0f}")
    print("=" * 60)
