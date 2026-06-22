"""Continuous lifelong learning: no frozen weights, every step learns.
Single loop with curriculum changes. Brain never stops learning.

Runs 6000 total steps with curriculum that changes every 500 steps:
  Phase 0 → 0 → 1 → 2 → 3 → 5 → 0 → 1 → 2 → 3 → 5 → 0
(Phases 0 and 5 appear 3 times each to test retention)

Metrics: per-100-step success rate, adaptation speed, retention.
"""

import torch, torch.nn.functional as F
from torch.distributions import Normal
from pathlib import Path
from env_nav import NavArena
from train_sr import (Hippocampus, Policy, CerebellarModel,
                      dopamine_update, PatternSeparator, DEVICE, ACC)
import logging, time, numpy as np

OUT = Path('results/continuous')
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'continuous.log', mode='w'), logging.StreamHandler()]
)
logger = logging.getLogger('continuous')


def run():
    logger.info("═" * 70)
    logger.info("PERCEPTA: CONTINUOUS LIFELONG LEARNING")
    logger.info("No frozen weights. Every step learns. Brain never stops.")
    logger.info("═" * 70)

    # ── Base training on demos ─────────────────────────────────
    from train_sr import train as train_base
    hc, pi, raw_fm = train_base(2000)
    logger.info(f"Base done: {len(hc)} patterns in hippocampus")

    # ── Optimizers for lifelong learning ───────────────────────
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)

    env = NavArena(render_mode=None)

    # ── Curriculum schedule ────────────────────────────────────
    # Each phase repeats to test retention
    curriculum = [
        (0, "Phase0-Fixed"),
        (1, "Phase1-RandStart"),
        (2, "Phase2-RandGoal"),
        (3, "Phase3-Walls"),
        (5, "Phase5-ProperMaze"),
        (0, "Phase0-Retention1"),
        (1, "Phase1-Retention"),
        (2, "Phase2-Retention"),
        (3, "Phase3-Retention"),
        (5, "Phase5-Retention"),
        (0, "Phase0-Retention2"),
        (5, "Phase5-Final"),
    ]
    steps_per_phase = 500
    total_steps = len(curriculum) * steps_per_phase

    # ── Tracking ───────────────────────────────────────────────
    goals = 0
    ep_s, ep_a, ep_r, ep_ns = [], [], [], []
    current_episode_id = len(hc.ca3.patterns)  # continue from base training
    dopamine_boost = 1.0
    dopamine_decay_steps = 0
    acc = ACC()

    # Per-100-step metrics
    step_goals = []  # goals in each 100-step window
    goals_this_window = 0
    phase_start_step = 0

    t0 = time.time()

    for phase_idx, (phase_id, phase_name) in enumerate(curriculum):
        env.set_curriculum(phase_id)
        s = env.reset(seed=None)[0]['state']
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE {phase_idx}: {phase_name} (env phase {phase_id})")
        logger.info(f"  Hippocampus: {len(hc)} patterns, SchemaBank: {len(hc.schema)} prototypes")

        phase_goals = 0

        for step_in_phase in range(steps_per_phase):
            global_step = phase_idx * steps_per_phase + step_in_phase
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

            # ── Action selection (hierarchical) ────────────────
            m, sd, v = pi(st)
            schema_action, confidence = hc.retrieve_actions(st.squeeze(0), k=10)
            traj_indices = hc.ca3.retrieve_trajectory(
                hc.dg(st.squeeze(0).unsqueeze(0)), k_steps=3)
            if traj_indices and len(traj_indices) > 0:
                action = hc.ca3.actions[traj_indices[0]].to(DEVICE).unsqueeze(0)
            elif schema_action is not None and confidence > 0.3:
                action = schema_action.unsqueeze(0)
            else:
                action = Normal(m, sd).sample()

            # ── Execute ────────────────────────────────────────
            obs2, re, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            s2 = obs2['state']
            done = term or trunc
            s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

            ep_s.append(st.squeeze(0).cpu())
            ep_a.append(action.squeeze(0).cpu())
            ep_r.append(re)
            ep_ns.append(s2_t.squeeze(0).cpu())

            # ── Cerebellar forward + hippocampal curiosity ─────
            sp, rp = raw_fm(st, action)
            if len(hc.ca3) > 0:
                Z = hc.ca3._get_Z(st.device)
                sims = torch.softmax(hc.dg(st.squeeze(0).unsqueeze(0)) @ Z.T * 5.0, dim=-1)
                novelty = 1.0 - sims.max().item()
            else:
                novelty = 1.0

            # ── Dopamine REINFORCE with curiosity ──────────────
            total_reward = re + 0.1 * novelty
            delta, lr_scale = dopamine_update(pi, opt_pi, opt_val, st, action,
                                              total_reward, s2_t,
                                              dopamine_boost=dopamine_boost)

            # ── Store in hippocampus ────────────────────────────
            hc.store(st.squeeze(0), action.squeeze(0), re, s2_t.squeeze(0),
                     episode_id=current_episode_id)

            # ── Cerebellar online learning ──────────────────────
            loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(
                rp.squeeze(-1), torch.tensor(re, device=DEVICE))
            opt_raw_fm.zero_grad(); loss_raw.backward()
            torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()

            # ── ACC stuck detection ─────────────────────────────
            vel = np.linalg.norm(s[2:4])
            stuck = vel < 0.05 and np.linalg.norm(
                action.squeeze(0).cpu().numpy()) > 0.5
            if stuck and not term:
                action = Normal(m, sd).sample() * 1.5

            # ── Dopamine decay ─────────────────────────────────
            if dopamine_decay_steps > 0:
                dopamine_decay_steps -= 1
                dopamine_boost = 1.0 + 4.0 * (dopamine_decay_steps / 25.0)
            else:
                dopamine_boost = 1.0

            # ── Goal reached ────────────────────────────────────
            if term:
                goals += 1
                phase_goals += 1
                goals_this_window += 1
                dopamine_boost = 5.0
                dopamine_decay_steps = 25
                if len(ep_s) > 1:
                    for i in range(len(ep_s)):
                        hc.store(ep_s[i], ep_a[i], ep_r[i], ep_ns[i],
                                 episode_id=current_episode_id)
                    ec_s = torch.stack(ep_s).to(DEVICE)
                    ec_a = torch.stack(ep_a).to(DEVICE)
                    for _ in range(30):
                        m_ec, _, _ = pi(ec_s)
                        loss_ec = F.mse_loss(m_ec, ec_a)
                        opt_pi.zero_grad(); loss_ec.backward()
                        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()

            # ── Episode end ─────────────────────────────────────
            if done:
                current_episode_id += 1
                s = env.reset(seed=None)[0]['state']
                ep_s, ep_a, ep_r, ep_ns = [], [], [], []
            else:
                s = s2

            # ── Sleep + SchemaBank update every 200 steps ───────
            if global_step > 0 and global_step % 200 == 0 and len(hc) >= 50:
                all_idx = list(range(len(hc)))
                hs, ha, hr, hn, hg = hc.get_batch(all_idx)
                hs, ha, hn = hs.to(DEVICE), ha.to(DEVICE), hn.to(DEVICE)
                hr, hg = hr.to(DEVICE), hg.to(DEVICE)

                # Structured RawFM replay (temporal order)
                init_sp, _ = raw_fm(hs, ha)
                init_loss = F.mse_loss(init_sp, hn).item()
                ep_ids = list(hc.ca3.episode_trajs.keys())
                bs = min(128, len(hn) // 10)
                for _ in range(1000):
                    if ep_ids:
                        ep_id = ep_ids[np.random.randint(len(ep_ids))]
                        traj = hc.ca3.episode_trajs[ep_id]
                        start = np.random.randint(0, max(1, len(traj) - bs))
                        seg = traj[start:min(start + bs, len(traj))]
                        ss = torch.stack([hc.ca3.states[i] for i in seg]).to(DEVICE)
                        sa = torch.stack([hc.ca3.actions[i] for i in seg]).to(DEVICE)
                        sn = torch.stack([hc.ca3.next_states[i] for i in seg]).to(DEVICE)
                        sr = torch.tensor([hc.ca3.rewards[i] for i in seg]).to(DEVICE)
                        sp_, rp_ = raw_fm(ss, sa)
                        loss = F.mse_loss(sp_, sn) + F.mse_loss(rp_.squeeze(-1), sr)
                    else:
                        idx = torch.randperm(len(hs), device=DEVICE)[:bs]
                        sp_, rp_ = raw_fm(hs[idx], ha[idx])
                        loss = F.mse_loss(sp_, hn[idx]) + F.mse_loss(rp_.squeeze(-1), hr[idx])
                    opt_raw_fm.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
                final_sp, _ = raw_fm(hs, ha)
                final_loss = F.mse_loss(final_sp, hn).item()
                logger.info(f"  Sleep: RawFM {init_loss:.4f}→{final_loss:.4f} ({len(hn)} trans)")

                # SchemaBank update (compression without deletion)
                hc.schema.update_from_ca3(hc, logger=logger)

                # Sleep BC with EPISODE-LEVEL SAMPLING (each episode gets equal weight)
                # Prevents catastrophic forgetting by ensuring old episodes are replayed
                # as often as new ones. The brain replays complete episodes, not transitions.
                all_ep_ids = sorted(hc.ca3.episode_trajs.keys())
                if len(all_ep_ids) > 0:
                    bc_states, bc_actions = [], []
                    n_per_ep = max(1, 500 // len(all_ep_ids))
                    for ep_id in all_ep_ids:
                        traj = hc.ca3.episode_trajs[ep_id]
                        k_samp = min(n_per_ep, len(traj))
                        idx = np.random.choice(traj, k_samp, replace=False)
                        bc_states.extend([hc.ca3.states[i] for i in idx])
                        bc_actions.extend([hc.ca3.actions[i] for i in idx])
                    if bc_states:
                        bs = torch.stack(bc_states).to(DEVICE)
                        ba = torch.stack(bc_actions).to(DEVICE)
                        bc_losses = []
                        for _ in range(50):
                            m_bc, _, _ = pi(bs)
                            loss_bc = F.mse_loss(m_bc, ba)
                            bc_losses.append(loss_bc.item())
                            opt_pi.zero_grad(); loss_bc.backward()
                            torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                        logger.info(f"  Sleep BC: {bc_losses[0]:.4f}→{bc_losses[-1]:.4f} "
                                    f"({len(bs)} trans, {n_per_ep}/ep from {len(all_ep_ids)} eps)")

            # ── Per-100-step logging ───────────────────────────
            if global_step > 0 and global_step % 100 == 0:
                phase_progress = global_step - phase_start_step
                logger.info(
                    f"  [{phase_name}] step={phase_progress:3d}/{steps_per_phase} "
                    f"goals={phase_goals} vel={vel:.1f} "
                    f"RPE={delta:+.2f} lr={lr_scale:.2f} "
                    f"hc={len(hc)} schema={len(hc.schema)}"
                )

        # ── End of phase ───────────────────────────────────────
        phase_start_step = global_step + 1

        if phase_goals > 0:
            avg_steps_per_goal = steps_per_phase / phase_goals
        else:
            avg_steps_per_goal = float('inf')
        logger.info(f"  → Phase {phase_name}: {phase_goals} goals in {steps_per_phase} steps "
                    f"({avg_steps_per_goal:.0f} steps/goal)")

    elapsed = time.time() - t0
    total_steps_run = len(curriculum) * steps_per_phase
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE: {goals} goals in {total_steps_run} steps ({elapsed:.1f}s)")
    logger.info(f"Final hippocampus: {len(hc)} patterns, SchemaBank: {len(hc.schema)} prototypes")

    # ── Retention analysis ────────────────────────────────────
    print("\n" + "=" * 70)
    print("RETENTION ANALYSIS: Same phase, repeated later")
    print("=" * 70)
    for phase_id, phase_label in [(0, "Phase0"), (1, "Phase1"), (2, "Phase2"),
                                    (3, "Phase3"), (5, "Phase5")]:
        appearances = [(i, name) for i, (pid, name) in enumerate(curriculum) if pid == phase_id]
        if len(appearances) >= 2:
            print(f"\n{phase_label} appears at indices: {[a[0] for a in appearances]}")
            print(f"  Retention: compare goal rates across appearances")

    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Avg step time: {elapsed/total_steps_run*1000:.1f}ms")


if __name__ == '__main__':
    run()
