"""Comprehensive generalization test: 4 phases, per-episode tracking.
Tests whether the agent adapts to random start/goal/maze via memory TD updates.
No catastrophic forgetting check: returns to phase 0 at the end.
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from hopfield_memory import PatternSeparator
from env_nav import NavArena
from train_sr import *
import json, time

OUT = Path('results/test_gen')
OUT.mkdir(parents=True, exist_ok=True)


def load_model():
    dg = PatternSeparator(2, 2000, 0.02)
    sf = SRNet().to(DEVICE)
    wrk = Worker().to(DEVICE)
    wrk.load_state_dict(torch.load(OUT.parent / 'sr' / 'worker.pt', map_location=DEVICE))
    mgr = Manager().to(DEVICE)
    mgr.load_state_dict(torch.load(OUT.parent / 'sr' / 'manager.pt', map_location=DEVICE))
    fm = nn.Sequential(nn.Linear(68, 128), nn.ReLU(), nn.Linear(128, 64)).to(DEVICE)
    fm.load_state_dict(torch.load(OUT.parent / 'sr' / 'fm.pt', map_location=DEVICE))

    mem = QMemory()
    sd = torch.load(OUT.parent / 'sr' / 'memory.pt', map_location='cpu')
    mem.keys = sd['keys']; mem.states = sd['states']; mem.actions = sd['actions']
    mem.qs = list(sd['qs']); mem.imp = list(sd['imp'])
    return dg, sf, wrk, mgr, fm, mem


def run_phase(env, dg, sf, wrk, mgr, fm, mem, phase_id, n_episodes,
              name="", track_mem=False):
    """Run episodes with memory TD updates enabled. Tracks all metrics."""
    env.set_curriculum(phase_id)
    results = {
        'name': name, 'phase': phase_id, 'n': n_episodes,
        'goals': 0, 'steps_to_goal': [], 'path_lengths': [],
        'q_values': [], 'td_errors': [], 'mem_sizes': [],
        'episode_paths': [],
    }

    for ep in range(n_episodes):
        s = env.reset(seed=42 if phase_id == 0 else None)[0]['state']
        hw = None; sg = torch.zeros(2, device=DEVICE); sgst = 0
        path = [s[:2].copy()]
        reached = False

        for t in range(500):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
            gd = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)

            if sgst % 10 == 0 or hw is None:
                sg, _, hw = mgr(st, gd, hw)

            with torch.no_grad():
                cand = mem.retrieve_topk(dg(st[:, :2]), 5)
                if cand is not None and len(mem) > 10:
                    phi_s = sf.phi(st)
                    gd_e = gd.expand(5, -1); phi_e = phi_s.expand(5, -1)
                    phi_n = fm(torch.cat([phi_e, cand, gd_e], -1))
                    a = cand[sf.q(phi_n).argmax().item()].unsqueeze(0)
                else:
                    m, sd, _, hw = wrk(st, gd, sg, hw)
                    a = Normal(m, sd).sample()

            s_next, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            s_next_state = s_next['state']
            path.append(s_next_state[:2].copy())
            sgst += 1

            # Memory TD update + storage (fast system adaptation)
            with torch.no_grad():
                st_next = torch.from_numpy(s_next_state).float().to(DEVICE).unsqueeze(0)
                q_next = sf.q(sf.phi(st_next)).item()
                q_now = sf.q(sf.phi(st)).item()
                td_target = re + 0.99 * q_next
                td_error = td_target - q_now
                results['td_errors'].append(td_error)
                results['q_values'].append(q_now)

                # Store new patterns at novel positions (enables adaptation)
                should_store = False
                if len(mem) < 100:
                    should_store = True  # bootstrap: fill memory initially
                elif len(mem) > 0:
                    # Check position novelty against stored states
                    pos = st[:, :2]
                    stored_positions = torch.stack([st_.unsqueeze(0) for st_ in mem.states[:50]]).to(pos.device)
                    stored_positions = stored_positions.squeeze(1)[:, :2]
                    dists = (pos - stored_positions).norm(dim=-1)
                    should_store = dists.min().item() > 0.5  # novel if > 0.5 units from any stored
                if should_store and abs(re + 0.99*q_next) < 200:
                    mem_goal_dir = (st[:, 2:4] - st[:, :2]) / ((st[:, 2:4] - st[:, :2]).norm(dim=-1, keepdim=True) + 1e-8)
                    mem.store(dg(st[:, :2]).squeeze(0), st.squeeze(0),
                             a.squeeze(0), re + 0.99 * q_next)

            s = s_next_state
            if term:
                reached = True
                break
            if trunc:
                hw = None; sgst = 0

        if reached:
            results['goals'] += 1
            results['steps_to_goal'].append(t)
            results['path_lengths'].append(len(path) - 1)

        results['episode_paths'].append(path)
        results['mem_sizes'].append(len(mem))

        if (ep + 1) % 10 == 0:
            rate = results['goals'] / (ep + 1)
            avg_steps = np.mean(results['steps_to_goal']) if results['steps_to_goal'] else 0
            print(f"    {ep+1:3d}/{n_episodes}: goal_rate={rate:.0%} avg_steps={avg_steps:.0f} mem={len(mem)}")

    return results


def main():
    print("=" * 60)
    print("GENERALIZATION TEST: 4 Phases + Catastrophic Forgetting Check")
    print("=" * 60)

    dg, sf, wrk, mgr, fm, mem = load_model()
    print(f"Loaded model + memory ({len(mem)} patterns)")

    env = NavArena(render_mode=None)
    all_results = []

    # Phase 0: Fixed start, fixed goal (baseline)
    print(f"\n{'='*60}\nPHASE 0: Fixed start → Fixed goal (BASELINE)")
    print(f"{'='*60}")
    r0 = run_phase(env, dg, sf, wrk, mgr, fm, mem, 0, 50, "fixed_fixed")
    all_results.append(r0)
    print(f"  Result: {r0['goals']}/50 = {r0['goals']/50:.0%}, "
          f"avg_steps={np.mean(r0['steps_to_goal']):.1f}")

    # Phase 1: Random start, fixed goal
    print(f"\n{'='*60}\nPHASE 1: Random start → Fixed goal")
    print(f"{'='*60}")
    r1 = run_phase(env, dg, sf, wrk, mgr, fm, mem, 1, 50, "random_fixed")
    all_results.append(r1)
    print(f"  Result: {r1['goals']}/50 = {r1['goals']/50:.0%}, "
          f"avg_steps={np.mean(r1['steps_to_goal']):.1f}")

    # Phase 2: Random start, random goal
    print(f"\n{'='*60}\nPHASE 2: Random start → Random goal")
    print(f"{'='*60}")
    r2 = run_phase(env, dg, sf, wrk, mgr, fm, mem, 2, 50, "random_random")
    all_results.append(r2)
    print(f"  Result: {r2['goals']}/50 = {r2['goals']/50:.0%}, "
          f"avg_steps={np.mean(r2['steps_to_goal']):.1f}")

    # Phase 3: Random start, random goal, random maze
    print(f"\n{'='*60}\nPHASE 3: Random start → Random goal → Random maze")
    print(f"{'='*60}")
    r3 = run_phase(env, dg, sf, wrk, mgr, fm, mem, 3, 50, "maze")
    all_results.append(r3)
    print(f"  Result: {r3['goals']}/50 = {r3['goals']/50:.0%}, "
          f"avg_steps={np.mean(r3['steps_to_goal']):.1f}")

    # Phase 4: BACK to fixed start, fixed goal (catastrophic forgetting check)
    print(f"\n{'='*60}\nPHASE 4: BACK to Fixed start → Fixed goal (FORGETTING CHECK)")
    print(f"{'='*60}")
    r4 = run_phase(env, dg, sf, wrk, mgr, fm, mem, 0, 50, "back_to_fixed")
    all_results.append(r4)
    print(f"  Result: {r4['goals']}/50 = {r4['goals']/50:.0%}, "
          f"avg_steps={np.mean(r4['steps_to_goal']):.1f}")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Phase':<45} {'GoalRate':>10} {'AvgSteps':>10} {'MemEnd':>8}")
    print(f"{'-'*75}")
    for r in all_results:
        avg_s = np.mean(r['steps_to_goal']) if r['steps_to_goal'] else 0
        print(f"  {r['name']:<43} {r['goals']/r['n']:>9.0%} {avg_s:>9.1f} {r['mem_sizes'][-1]:>8}")

    # Catastrophic forgetting check
    if r4['goals'] / 50 >= r0['goals'] / 50 * 0.8:
        print(f"\n  ✅ No catastrophic forgetting: Phase 4 ({r4['goals']/50:.0%}) >= 80% of Phase 0 ({r0['goals']/50:.0%})")
    else:
        print(f"\n  ⚠️ Some forgetting detected: Phase 4 ({r4['goals']/50:.0%}) < 80% of Phase 0 ({r0['goals']/50:.0%})")

    # Save results
    summary = {r['name']: {'goal_rate': r['goals']/r['n'],
                           'avg_steps': float(np.mean(r['steps_to_goal'])) if r['steps_to_goal'] else 0,
                           'mem_end': r['mem_sizes'][-1]} for r in all_results}
    print(f"\n  Results saved to {OUT / 'summary.json'}")
    with open(OUT / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
