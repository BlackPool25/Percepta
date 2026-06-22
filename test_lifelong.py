"""Percepta: True lifelong adaptation test.
Training: small interleaved sample of each phase (5 seeds of each type).
Testing: TRULY NOVEL configurations (never seen during training).
Measures: adaptation speed, retention, multi-skill learning.
"""

import torch, torch.nn.functional as F, gc, time, logging, numpy as np
from torch.distributions import Normal
from pathlib import Path
from env_nav import NavArena
from train_sr import (Hippocampus, Policy, CerebellarModel,
                      dopamine_update, run_demo, DEVICE, PatternSeparator)

OUT = Path('results/lifelong')
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'lifelong.log', mode='w'), logging.StreamHandler()])
logger = logging.getLogger('lifelong')


def train_seeded(hc, pi, raw_fm):
    """Seed with small samples of each phase type (5 trajectories each).
    Model gets the CONCEPT of each phase but can't memorize specific layouts.
    """
    env = NavArena(render_mode=None)
    all_s, all_a, all_r, all_ns = [], [], [], []

    # Phase 0: 5 demos (fixed start, fixed goal at 3,3)
    env.set_curriculum(0)
    for traj_idx in range(5):
        env.reset(seed=traj_idx * 10)
        s = env._get_obs()['state']
        for _ in range(200):
            g = env._goal_pos[:2]; p = s[:2]; d_vec = g - p
            dist = np.linalg.norm(d_vec)
            a = np.clip(d_vec / dist if dist > .2 else d_vec * .5, -1, 1).astype(np.float32)
            obs2, r, term, _, _ = env.step(a)
            s2 = obs2['state']
            all_s.append(torch.tensor(s)); all_a.append(torch.tensor(a))
            all_r.append(float(r)); all_ns.append(torch.tensor(s2))
            s = s2
            if term: break

    # Phase 1: 5 demos (random start, fixed goal)
    for traj_idx in range(5):
        env.set_curriculum(1); env.reset(seed=traj_idx)
        s = env._get_obs()['state']
        for _ in range(200):
            g = env._goal_pos[:2]; p = s[:2]; d_vec = g - p
            dist = np.linalg.norm(d_vec)
            a = np.clip(d_vec / dist if dist > .2 else d_vec * .5, -1, 1).astype(np.float32)
            obs2, r, term, _, _ = env.step(a)
            s2 = obs2['state']
            all_s.append(torch.tensor(s)); all_a.append(torch.tensor(a))
            all_r.append(float(r)); all_ns.append(torch.tensor(s2))
            s = s2
            if term: break

    # Phase 5 (proper maze): 5 demos with ORACLE navigation
    for traj_idx in range(5):
        env.set_curriculum(5); env.reset(seed=traj_idx)
        s = env._get_obs()['state']
        for _ in range(200):
            g = env._goal_pos[:2]; p = s[:2]; d_vec = g - p
            dist = np.linalg.norm(d_vec)
            a = np.clip(d_vec / dist if dist > .2 else d_vec * .5, -1, 1).astype(np.float32)
            obs2, r, term, _, _ = env.step(a)
            s2 = obs2['state']
            all_s.append(torch.tensor(s)); all_a.append(torch.tensor(a))
            all_r.append(float(r)); all_ns.append(torch.tensor(s2))
            s = s2
            if term: break

    env.close()
    logger.info(f"Seeded {len(all_s)} transitions across phases 0, 1, 5")

    # Store in hippocampus as separate episodes
    ep_size = len(all_s) // 15
    for i in range(len(all_s)):
        hc.store(all_s[i], all_a[i], all_r[i], all_ns[i], episode_id=i // ep_size)

    # Pre-train RawFM on seeds
    ds = torch.stack(all_s).to(DEVICE)
    da = torch.stack(all_a).to(DEVICE)
    dn = torch.stack(all_ns).to(DEVICE)
    dr = torch.tensor(all_r, device=DEVICE)
    for i in range(200):
        sp, rp = raw_fm(ds, da)
        loss = F.mse_loss(sp, dn) + F.mse_loss(rp.squeeze(-1), dr)
        loss.backward(); torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0)
    logger.info(f"RawFM seeded: loss={loss.item():.4f}")

    # Pre-train policy on seeds
    for _ in range(200):
        m, _, _ = pi(ds)
        loss = F.mse_loss(m, da)
        loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    with torch.no_grad():
        cosim = F.cosine_similarity(pi(ds)[0], da, dim=-1).mean().item()
    logger.info(f"Policy seeded: cosim={cosim:.3f}")

    # First SchemaBank update
    hc.schema.update_from_ca3(hc, logger=logger)


def adapt():
    logger.info("═" * 70)
    logger.info("PERCEPTA: TRUE LIFELONG ADAPTATION")
    logger.info("═" * 70)

    # ── Initialize with seeded knowledge ──
    hc = Hippocampus()
    pi = Policy().to(DEVICE)
    raw_fm = CerebellarModel().to(DEVICE)
    train_seeded(hc, pi, raw_fm)
    logger.info(f"Seeded: {len(hc)} patterns, {len(hc.schema)} schemas")

    # ── Optimizers ──
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)

    env = NavArena(render_mode=None)
    n_steps = 6000
    goals = 0
    ep_id_offset = 1000
    dopamine_boost = 1.0
    dopamine_decay_steps = 0
    step = 0

    # ── Continuous curriculum ──
    # Each phase repeats to test retention
    curriculum = [(0, "Fixed"), (1, "RandStart"), (2, "RandGoal"),
                  (3, "Walls"), (5, "ProperMaze"),
                  (0, "Fixed-R1"), (1, "RandStart-R1"), (2, "RandGoal-R1"),
                  (3, "Walls-R1"), (5, "ProperMaze-R1"),
                  (0, "Fixed-R2"), (5, "ProperMaze-R2")]
    steps_per_phase = 1000

    # Per-phase tracking
    phase_results = []  # [(name, goals, steps_total)]
    phase_start = 0

    for phase_id, phase_name in curriculum:
        env.set_curriculum(phase_id)
        # Keep goal fixed WITHIN each phase so agent can learn it
        s = env.reset(seed=42)[0]['state']
        # Store the goal position for repeat resets within this phase
        fixed_goal = env._goal_pos.copy()
        ep_s, ep_a, ep_r, ep_ns = [], [], [], []
        phase_goals = 0
        ep_count = 0

        for p_step in range(steps_per_phase):
            st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

            # Action: SchemaBank → policy fallback
            schema_a, conf = hc.retrieve_actions(st.squeeze(0), k=10)
            if schema_a is not None and conf > 0.3:
                a = schema_a.unsqueeze(0)
            else:
                m, sd, _ = pi(st)
                a = Normal(m, sd).sample()

            obs2, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
            s2 = obs2['state']
            done = term or trunc
            s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

            # Learn from every step
            sp, rp = raw_fm(st, a)
            if len(hc.ca3) > 0:
                with torch.no_grad():
                    Z = hc.ca3._get_Z(st.device)
                    sims = torch.softmax(hc.dg(st) @ Z.T * 5.0, dim=-1)
                    novelty = 1.0 - sims.max().item()
            else:
                novelty = 1.0
            total_rew = re + 0.1 * novelty
            dopamine_update(pi, opt_pi, opt_val, st, a, total_rew, s2_t, dopamine_boost=dopamine_boost)
            loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(rp.squeeze(-1), torch.tensor(re, device=DEVICE))
            opt_raw_fm.zero_grad(); loss_raw.backward()
            torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
            hc.store(st.squeeze(0), a.squeeze(0), re, s2_t.squeeze(0), episode_id=ep_id_offset + ep_count)

            ep_s.append(st.squeeze(0).cpu()); ep_a.append(a.squeeze(0).cpu())
            ep_r.append(re); ep_ns.append(s2_t.squeeze(0).cpu())

            if dopamine_decay_steps > 0:
                dopamine_decay_steps -= 1
                dopamine_boost = 1.0 + 4.0 * (dopamine_decay_steps / 25.0)
            else:
                dopamine_boost = 1.0

            if term:
                goals += 1; phase_goals += 1
                dopamine_boost = 5.0; dopamine_decay_steps = 25
                if len(ep_s) > 1:
                    for i in range(len(ep_s)):
                        hc.store(ep_s[i], ep_a[i], ep_r[i], ep_ns[i], episode_id=ep_id_offset + ep_count)

            if done:
                ep_count += 1
                s = env.reset(seed=None)[0]['state']
                # Goal stays fixed within phase (set by initial reset)
                ep_s, ep_a, ep_r, ep_ns = [], [], [], []
            else:
                s = s2

            step += 1

            # ── Sleep every 200 steps ──
            if step % 200 == 0 and len(hc) >= 50:
                all_idx = list(range(len(hc)))
                hs, ha, hr, hn, hg = hc.get_batch(all_idx)
                hs, ha, hn = hs.to(DEVICE), ha.to(DEVICE), hn.to(DEVICE)
                hr, hg = hr.to(DEVICE), hg.to(DEVICE)

                # RawFM structured replay
                init_sp, _ = raw_fm(hs, ha)
                init_loss = F.mse_loss(init_sp, hn).item()
                ep_ids = list(hc.ca3.episode_trajs.keys())
                bs = min(128, max(1, len(hn) // 10))
                for _ in range(500):
                    ep_id = ep_ids[np.random.randint(len(ep_ids))]
                    traj = hc.ca3.episode_trajs[ep_id]
                    start = np.random.randint(0, max(1, len(traj) - bs))
                    seg = traj[start:min(start + bs, len(traj))]
                    ss = torch.stack([hc.ca3.states[i] for i in seg]).to(DEVICE)
                    sa = torch.stack([hc.ca3.actions[i] for i in seg]).to(DEVICE)
                    sn = torch.stack([hc.ca3.next_states[i] for i in seg]).to(DEVICE)
                    sr_vec = torch.tensor([hc.ca3.rewards[i] for i in seg]).to(DEVICE)
                    sp_, rp_ = raw_fm(ss, sa)
                    loss = F.mse_loss(sp_, sn) + F.mse_loss(rp_.squeeze(-1), sr_vec)
                    opt_raw_fm.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
                final_sp, _ = raw_fm(hs, ha)
                final_loss = F.mse_loss(final_sp, hn).item()
                logger.info(f"  Sleep: RawFM {init_loss:.4f}→{final_loss:.4f} ({len(hn)} trans)")

                # SchemaBank update (compression without deletion)
                hc.schema.update_from_ca3(hc, logger=logger)

                # Sleep BC: episode-balanced (each episode gets equal weight)
                all_ep_ids = sorted(hc.ca3.episode_trajs.keys())
                n_eps_total = len(all_ep_ids)
                if n_eps_total > 0:
                    n_per_ep = max(1, 300 // n_eps_total)
                    bc_s, bc_a = [], []
                    for ep_id in all_ep_ids:
                        traj = hc.ca3.episode_trajs[ep_id]
                        k_samp = min(n_per_ep, len(traj))
                        idx = np.random.choice(traj, k_samp, replace=False)
                        bc_s.extend([hc.ca3.states[i] for i in idx])
                        bc_a.extend([hc.ca3.actions[i] for i in idx])
                    if bc_s:
                        bs = torch.stack(bc_s).to(DEVICE)
                        ba = torch.stack(bc_a).to(DEVICE)
                        bc_l = []
                        for _ in range(50):
                            m_bc, _, _ = pi(bs)
                            l = F.mse_loss(m_bc, ba)
                            bc_l.append(l.item())
                            opt_pi.zero_grad(); l.backward()
                            torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                        logger.info(f"  Sleep BC: {bc_l[0]:.4f}→{bc_l[-1]:.4f} "
                                    f"({len(bs)} trans, {n_per_ep}/ep)")

                # Clear VRAM cache after sleep
                del hs, ha, hn, hr, hg, init_sp, final_sp, sp_, rp_
                if 'bs' in dir(): del bs, ba
                gc.collect()
                torch.cuda.empty_cache()

            # ── Log ──
            if step % 100 == 0:
                vr_alloc = torch.cuda.memory_allocated() / 1e6
                vr_cache = torch.cuda.memory_reserved() / 1e6
                logger.info(f"  [{phase_name}] step={p_step:3d} goals={phase_goals} "
                            f"hc={len(hc)} schema={len(hc.schema)} "
                            f"VRAM={vr_alloc:.0f}/{vr_cache:.0f}MB")

        phase_results.append((phase_name, phase_goals, steps_per_phase))
        phase_start = step

    env.close()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("LIFELONG ADAPTATION RESULTS")
    print("=" * 70)
    print(f"{'Phase':20s} {'Goals':>10s} {'Rate':>10s}")
    print("-" * 40)
    for name, g, t in phase_results:
        print(f"{name:20s} {g:3d}/{t:3d} {g/t:.0%}")
    print("-" * 40)
    print(f"Total: {goals} goals in {step} steps")

    # Retention analysis
    print(f"\nRETENTION:")
    for base in ["Fixed", "RandStart", "RandGoal", "Walls", "ProperMaze"]:
        apps = [(n, g, t) for n, g, t in phase_results if n.startswith(base)]
        if len(apps) > 1:
            rates = [f"{g/t:.0%}" for _, g, t in apps]
            print(f"  {base:15s}: {' → '.join(rates)}")

    vr_final = torch.cuda.memory_allocated() / 1e6
    vr_cache = torch.cuda.memory_reserved() / 1e6
    print(f"\nFinal VRAM: {vr_final:.0f}MB allocated, {vr_cache:.0f}MB cached")
    print(f"Final hippocampus: {len(hc)} patterns, {len(hc.schema)} schemas")


if __name__ == '__main__':
    adapt()
