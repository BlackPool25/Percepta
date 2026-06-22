"""Percepta: True curriculum learning.
Phase 1 - TEACH: small sample of ALL types to understand gist (not memorize)
Phase 2 - PRACTICE: each type practiced extensively in long blocks
Phase 3 - RETENTION: short tests of each type to measure true learning
"""

import torch, torch.nn.functional as F, gc, time, logging, numpy as np
from torch.distributions import Normal
from pathlib import Path
from env_nav import NavArena
from train_sr import (Hippocampus, Policy, CerebellarModel,
                      dopamine_update, run_demo, DEVICE)

OUT = Path('results/curriculum')
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'curriculum.log', mode='w'), logging.StreamHandler()])
logger = logging.getLogger('curriculum')


def teach_gist(hc, pi, raw_fm):
    """Phase 1: Show agent small samples of each phase type to build gist understanding.
    Only 5 trajectories per type — enough to understand, not enough to memorize.
    """
    env = NavArena(render_mode=None)
    all_s, all_a, all_r, all_ns = [], [], [], []

    for phase_id, label in [(0, "Fixed"), (1, "RandStart"), (2, "RandGoal"),
                             (3, "Walls"), (5, "ProperMaze")]:
        env.set_curriculum(phase_id)
        for traj_idx in range(5):
            env.reset(seed=traj_idx * 10 + phase_id)
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
        logger.info(f"  {label}: {len(all_s)} total transitions so far")

    env.close()
    logger.info(f"Gist seeded: {len(all_s)} transitions across phases 0,1,2,3,5")

    ep_size = max(1, len(all_s) // 25)
    for i in range(len(all_s)):
        hc.store(all_s[i], all_a[i], all_r[i], all_ns[i], episode_id=i // ep_size)

    ds = torch.stack(all_s).to(DEVICE)
    da = torch.stack(all_a).to(DEVICE)
    dn = torch.stack(all_ns).to(DEVICE)
    dr = torch.tensor(all_r, device=DEVICE)
    opt_raw = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)
    for i in range(500):
        sp, rp = raw_fm(ds, da)
        loss = F.mse_loss(sp, dn) + F.mse_loss(rp.squeeze(-1), dr)
        opt_raw.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw.step()
    logger.info(f"RawFM gist: loss={loss.item():.4f}")
    opt_pi_init = torch.optim.Adam(pi.parameters(), lr=1e-3)
    for _ in range(500):
        m, _, _ = pi(ds)
        loss = F.mse_loss(m, da)
        opt_pi_init.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi_init.step()
    with torch.no_grad():
        cosim = F.cosine_similarity(pi(ds)[0], da, dim=-1).mean().item()
    logger.info(f"Policy gist: cosim={cosim:.3f}")
    hc.schema.update_from_ca3(hc, logger=logger)


def practice_phase(env, hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm,
                   phase_id, n_steps, label, step_counter):
    """Practice one phase type extensively. Goal stays fixed within phase."""
    env.set_curriculum(phase_id)
    s = env.reset(seed=42)[0]['state']
    goals, ep_s, ep_a, ep_r, ep_ns = 0, [], [], [], []
    ep_id = max(hc.ca3.episode_ids) + 1 if hc.ca3.episode_ids else 0
    dopamine_boost, dopamine_decay = 1.0, 0

    for p_step in range(n_steps):
        global_step = step_counter[0]
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

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

        sp, rp = raw_fm(st, a)
        with torch.no_grad():
            Z = hc.ca3._get_Z(st.device)
            sims = torch.softmax(hc.dg(st) @ Z.T * 5.0, dim=-1)
            novelty = 1.0 - sims.max().item()
        total_rew = re + 0.1 * novelty
        dopamine_update(pi, opt_pi, opt_val, st, a, total_rew, s2_t, dopamine_boost=dopamine_boost)
        loss_raw = F.mse_loss(sp, s2_t.detach()) + F.mse_loss(rp.squeeze(-1), torch.tensor(re, device=DEVICE))
        opt_raw_fm.zero_grad(); loss_raw.backward()
        torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
        hc.store(st.squeeze(0), a.squeeze(0), re, s2_t.squeeze(0), episode_id=ep_id)

        ep_s.append(st.squeeze(0).cpu()); ep_a.append(a.squeeze(0).cpu())
        ep_r.append(re); ep_ns.append(s2_t.squeeze(0).cpu())

        if dopamine_decay > 0:
            dopamine_decay -= 1
            dopamine_boost = 1.0 + 4.0 * (dopamine_decay / 25.0)
        else:
            dopamine_boost = 1.0

        if term:
            goals += 1
            dopamine_boost = 5.0; dopamine_decay = 25
            if len(ep_s) > 1:
                for i in range(len(ep_s)):
                    hc.store(ep_s[i], ep_a[i], ep_r[i], ep_ns[i], episode_id=ep_id)

        if done:
            ep_id += 1
            s = env.reset(seed=None)[0]['state']
            ep_s, ep_a, ep_r, ep_ns = [], [], [], []
        else:
            s = s2

        step_counter[0] += 1
        global_step = step_counter[0]

        if global_step % 400 == 0:
            vr = torch.cuda.memory_allocated() / 1e6
            logger.info(f"  [{label}] step={p_step:4d}/{n_steps} goals={goals} "
                        f"hc={len(hc)} schema={len(hc.schema)} VRAM={vr:.0f}MB")

        if global_step % 800 == 0:
            all_idx = list(range(len(hc)))
            hs, ha, hr, hn, hg = hc.get_batch(all_idx)
            hs, ha, hn = hs.to(DEVICE), ha.to(DEVICE), hn.to(DEVICE)
            hr, hg = hr.to(DEVICE), hg.to(DEVICE)
            init_loss = F.mse_loss(raw_fm(hs, ha)[0], hn).item()
            ep_ids = list(hc.ca3.episode_trajs.keys())
            bs = min(128, max(1, len(hn) // 10))
            for _ in range(500):
                eid = ep_ids[np.random.randint(len(ep_ids))]
                traj = hc.ca3.episode_trajs[eid]
                start = np.random.randint(0, max(1, len(traj) - bs))
                seg = traj[start:min(start + bs, len(traj))]
                ss = torch.stack([hc.ca3.states[i] for i in seg]).to(DEVICE)
                sa = torch.stack([hc.ca3.actions[i] for i in seg]).to(DEVICE)
                sn = torch.stack([hc.ca3.next_states[i] for i in seg]).to(DEVICE)
                sr = torch.tensor([hc.ca3.rewards[i] for i in seg]).to(DEVICE)
                sp_, rp_ = raw_fm(ss, sa)
                loss = F.mse_loss(sp_, sn) + F.mse_loss(rp_.squeeze(-1), sr)
                opt_raw_fm.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_fm.parameters(), 1.0); opt_raw_fm.step()
            final_loss = F.mse_loss(raw_fm(hs, ha)[0], hn).item()
            logger.info(f"  Sleep: RawFM {init_loss:.4f}→{final_loss:.4f} ({len(hs)} trans)")

            hc.schema.update_from_ca3(hc, logger=logger)

            all_ep_ids = sorted(hc.ca3.episode_trajs.keys())
            if all_ep_ids:
                n_per_ep = max(1, 500 // len(all_ep_ids))
                bc_s, bc_a = [], []
                for eid in all_ep_ids:
                    traj = hc.ca3.episode_trajs[eid]
                    k_s = min(n_per_ep, len(traj))
                    idx = np.random.choice(traj, k_s, replace=False)
                    bc_s.extend([hc.ca3.states[i] for i in idx])
                    bc_a.extend([hc.ca3.actions[i] for i in idx])
                if bc_s:
                    bs = torch.stack(bc_s).to(DEVICE)
                    ba = torch.stack(bc_a).to(DEVICE)
                    bc_l = []
                    for _ in range(50):
                        l = F.mse_loss(pi(bs)[0], ba)
                        bc_l.append(l.item())
                        opt_pi.zero_grad(); l.backward()
                        torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt_pi.step()
                    logger.info(f"  Sleep BC: {bc_l[0]:.4f}→{bc_l[-1]:.4f} ({len(bs)} trans)")

            del hs, ha, hn, hr, hg
            gc.collect(); torch.cuda.empty_cache()

    return goals


def test_retention(env, hc, pi, raw_fm, n_eps=20):
    """True test: each episode is a novel instance. Agent must adapt on the fly."""
    results = {}
    for phase_id, label in [(0, "Fixed"), (1, "RandStart"), (2, "RandGoal"),
                             (3, "Walls"), (5, "ProperMaze")]:
        env.set_curriculum(phase_id)
        goals = 0
        for ep in range(n_eps):
            s = env.reset(seed=None)[0]['state']
            for _ in range(500):
                st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
                schema_a, conf = hc.retrieve_actions(st.squeeze(0), k=10)
                if schema_a is not None and conf > 0.3:
                    a = schema_a.unsqueeze(0)
                else:
                    a = Normal(*pi(st)[:2]).sample()
                obs2, _, term, _, _ = env.step(a.squeeze(0).cpu().numpy())
                s = obs2['state']
                if term:
                    goals += 1; break
        results[label] = goals
        logger.info(f"  Retention {label}: {goals}/{n_eps} = {goals/n_eps:.0%}")
    return results


def main():
    logger.info("═" * 70)
    logger.info("PERCEPTA: CURRICULUM LEARNING")
    logger.info("═" * 70)

    hc = Hippocampus(); pi = Policy().to(DEVICE)
    raw_fm = CerebellarModel().to(DEVICE)
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)

    # ── PHASE 1: TEACH GIST ──
    logger.info("\n📚 PHASE 1: TEACH GIST (small samples of all types)")
    teach_gist(hc, pi, raw_fm)
    logger.info(f"  → {len(hc)} patterns, {len(hc.schema)} schemas")

    # ── PHASE 2: PRACTICE EACH TYPE ──
    logger.info("\n✏️  PHASE 2: PRACTICE (each type extensively)")
    practice_configs = [
        (0, "FixedGoal",     2000),
        (1, "RandStart",     2000),
        (2, "RandGoal",      2000),
        (3, "Walls",         2000),
        (5, "ProperMaze",    2000),
    ]
    step_counter = [0]
    env = NavArena(render_mode=None)

    practice_results = {}
    for phase_id, label, n_steps in practice_configs:
        logger.info(f"\n─── Practice: {label} ({n_steps} steps) ───")
        t0 = time.time()
        goals = practice_phase(env, hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm,
                                phase_id, n_steps, label, step_counter)
        elapsed = time.time() - t0
        practice_results[label] = goals
        logger.info(f"  → {goals} goals in {n_steps} steps ({elapsed:.1f}s)")

    env.close()

    # ── PHASE 3: TEST RETENTION ──
    env = NavArena(render_mode=None)
    logger.info("\n🧪 PHASE 3: RETENTION TEST")
    retention_results = test_retention(env, hc, pi, raw_fm, n_eps=20)
    env.close()

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING RESULTS")
    print("=" * 70)
    print(f"\n📚 Gist: {len(hc)} patterns in hippocampus, {len(hc.schema)} schemas")
    print(f"\n✏️  Practice:")
    for label, goals in practice_results.items():
        print(f"  {label:15s}: {goals} goals")
    print(f"\n🧪 Retention (zero-shot, 20 episodes each):")
    for label, goals in retention_results.items():
        print(f"  {label:15s}: {goals}/20 = {goals/20:.0%}")
    print(f"\n📊 Final: {len(hc)} patterns, {len(hc.schema)} schemas")
    vr = torch.cuda.memory_allocated() / 1e6
    vc = torch.cuda.memory_reserved() / 1e6
    print(f"   VRAM: {vr:.0f}MB / {vc:.0f}MB")
    print(f"   Steps: {step_counter[0]}")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
