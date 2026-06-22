"""Lifelong adaptation test: measures multi-skill learning without forgetting.
Trains once on Phase 0, then cycles through all phases MULTIPLE TIMES.
Same model, same hippocampus, continuous online adaptation.
Tests: retention, transfer, multi-skill mastery, memory scaling.

Phases (no goal in state — agent navigates from MEMORY):
  0: Fixed start → Fixed goal (baseline)
  1: Random start → Fixed goal
  2: Random start → Random goal
  3: Random start → Random goal + 4 wall configs
  5: Random start → Random goal + PROPER SOLVABLE MAZES (7x7)

Each cycle = all 5 phases sequentially. 3 cycles total = 15 phases.
Agent NEVER resets. Memory accumulates across ALL phases and cycles.
"""

import torch, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from env_nav import NavArena
from train_sr import (Hippocampus, Policy, CerebellarModel,
                      dopamine_update, DEVICE)
import logging, time, numpy as np

OUT = Path('results/adapt')
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'adapt.log', mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger('adapt_test')


def run_phase(env, hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm,
              phase_id, n_episodes=50, cycle=0, phase_in_cycle=0):
    """Run episodes with online adaptation. Returns (success_list, steps_list)."""
    env.set_curriculum(phase_id)
    ep_results, steps_to_goal = [], []
    hc_before = len(hc)

    for ep in range(n_episodes):
        s = env.reset(seed=None)[0]['state']
        reached = False
        ep_steps = 0

        for _ in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

            schema_action, confidence = hc.retrieve_actions(st.squeeze(0), k=10)
            if schema_action is not None and confidence > 0.3:
                a = schema_action  # shape (2,)
            else:
                m, sd, _ = pi(st)
                a = Normal(m, sd).sample().squeeze(0)  # shape (2,)

            obs2, re, term, trunc, _ = env.step(a.cpu().numpy())
            s2 = obs2['state']
            s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

            # Online adaptation: learn from every step
            sp, rp = raw_fm(st, a.unsqueeze(0))
            if len(hc.ca3) > 0:
                z_q = hc.dg(st.squeeze(0).unsqueeze(0))
                Z = hc.ca3._get_Z(st.device)
                sims = torch.softmax(z_q @ Z.T * 5.0, dim=-1)
                novelty = 1.0 - sims.max().item()
            else:
                novelty = 1.0
            total_rew = re + 0.1 * novelty
            dopamine_update(pi, opt_pi, opt_val, st, a.unsqueeze(0),
                            total_rew, s2_t, dopamine_boost=1.0)
            loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(
                rp.squeeze(-1), torch.tensor(re, device=DEVICE))
            opt_raw_fm.zero_grad(); loss_raw.backward()
            torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
            hc.store(st.squeeze(0), a.squeeze(0), re, s2_t.squeeze(0), episode_id=int(ep + cycle * 1000))

            s = s2
            ep_steps += 1
            if term:
                reached = True
                steps_to_goal.append(ep_steps)
                break

        ep_results.append(1 if reached else 0)

    hc_growth = len(hc) - hc_before
    return ep_results, steps_to_goal, hc_growth


def main():
    logger.info("═" * 70)
    logger.info("PERCEPTA: LIFELONG ADAPTATION TEST")
    logger.info("═" * 70)

    # ── Train once on simple Phase 0 ───────────────────────────
    from train_sr import train as train_base
    hc, pi, raw_fm = train_base(2000)
    logger.info(f"Base training: {len(hc)} patterns")

    # ── Optimizers for lifelong online adaptation ──────────────
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)

    env = NavArena(render_mode=None)

    # ── Phase definitions ──────────────────────────────────────
    phases = [
        (0, "FixedStart"),
        (1, "RandomStart"),
        (2, "RandomGoal"),
        (3, "Walls"),
        (5, "ProperMaze"),
    ]

    n_cycles = 3  # loop through all phases 3 times
    n_eps_per_phase = 30  # episodes per phase per cycle

    # Store results: phase_results[phase_name][cycle] = {'success': [...], 'steps': [...]}
    phase_results = {name: [] for _, name in phases}

    # ── Lifelong adaptation loop ───────────────────────────────
    for cycle in range(n_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE {cycle + 1}/{n_cycles}")
        logger.info(f"{'='*60}")

        for p_idx, (phase_id, name) in enumerate(phases):
            desc = {0: "Fixed start → Fixed goal",
                    1: "Random start → Fixed goal",
                    2: "Random start → Random goal",
                    3: "4 wall configs",
                    5: "Proper solvable 7×7 maze"}[phase_id]

            logger.info(f"\n── Cycle {cycle+1}, Phase {phase_id}: {desc} ──")
            t0 = time.time()

            ep_results, steps, hc_growth = run_phase(
                env, hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm,
                phase_id, n_episodes=n_eps_per_phase,
                cycle=cycle, phase_in_cycle=p_idx
            )

            elapsed = time.time() - t0
            total = sum(ep_results)
            avg_steps = np.mean(steps) if steps else 0

            # Log rolling performance (first 10, last 10)
            first10 = sum(ep_results[:10])
            last10 = sum(ep_results[-10:])
            logger.info(f"  → {total}/{n_eps_per_phase} = {total/n_eps_per_phase:.0%} "
                        f"(first10={first10}/10, last10={last10}/10, "
                        f"avg_steps={avg_steps:.0f}, hc+={hc_growth}, {elapsed:.1f}s)")

            phase_results.setdefault(name, []).append({
                'success': ep_results,
                'steps': steps,
                'hc_before': len(hc) - hc_growth,
                'hc_after': len(hc),
            })

    env.close()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"LIFELONG ADAPTATION SUMMARY ({n_cycles} cycles × {len(phases)} phases)")
    print("=" * 70)

    header = f"{'Phase':15s}"
    for c in range(n_cycles):
        header += f" {'Cycle'+str(c+1):>12s}"
    header += f" {'Retention':>12s}"
    print(header)
    print("-" * 70)

    for name in [n for _, n in phases]:
        row = f"{name:15s}"
        cycle_scores = []
        for c in range(n_cycles):
            s = phase_results[name][c]['success']
            score = sum(s) / len(s)
            cycle_scores.append(score)
            row += f" {score:.0%} ({sum(s):2d}/{len(s):2d})  "
        # Retention: score in cycle 3 vs cycle 1
        if len(cycle_scores) >= 2:
            retention = f"{cycle_scores[-1]:.0%} vs {cycle_scores[0]:.0%}"
        else:
            retention = "N/A"
        row += f" {retention:>12s}"
        print(row)

    print("-" * 70)

    # ── Memory scaling ─────────────────────────────────────────
    total_patterns = len(hc)
    print(f"\nMemory scaling: {total_patterns} total patterns across all phases")
    print(f"  Phase 0 baseline: ~{phase_results['FixedStart'][0]['hc_after']}")
    print(f"  Final ({n_cycles} cycles): {total_patterns}")

    # ── Multi-skill retention check ────────────────────────────
    print(f"\nMulti-skill retention (cycle 3 performance):")
    for name in [n for _, n in phases]:
        s = phase_results[name][-1]['success']
        print(f"  {name:15s}: {sum(s)}/{len(s)} = {sum(s)/len(s):.0%}")


if __name__ == '__main__':
    main()
