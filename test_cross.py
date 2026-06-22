"""Cross-environment retention: can the same agent learn multiple skills?
1. Train on standard maze tasks (Phase 0-5)
2. Train on Bizonal zone rules
3. Test retention of BOTH
4. Neither overwrites the other
"""

import torch, torch.nn.functional as F, gc, time, logging, numpy as np
from torch.distributions import Normal
from pathlib import Path
from env_nav import NavArena
from env_bizonal import BizonalArena
from train_sr import (Hippocampus, Policy, CerebellarModel,
                      dopamine_update, run_demo, DEVICE)

OUT = Path('results/cross')
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'cross.log', mode='w'), logging.StreamHandler()])
logger = logging.getLogger('cross')


def test_maze(env, hc, pi, n_eps=10):
    """Standard maze retention test."""
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
                a = schema_a.unsqueeze(0) if (schema_a is not None and conf > 0.3) else Normal(*pi(st)[:2]).sample()
                obs2, _, term, _, _ = env.step(a.squeeze(0).cpu().numpy())
                s = obs2['state']
                if term:
                    goals += 1; break
        results[label] = goals
    return results


def test_bizonal(env, hc, pi, n_eps=10):
    """Bizonal retention test. Returns (LEFT_success, RIGHT_success, total)."""
    left_g, right_g, wrong = 0, 0, 0
    for side, label in [(-1, "L"), (1, "R")]:
        for ep in range(n_eps):
            obs, _ = env.reset(seed=ep * 10 + 42)
            start_x = env.data.xpos[env._agent_body_id][0]
            if start_x * side < 0: continue
            s = obs
            for _ in range(500):
                st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
                schema_a, conf = hc.retrieve_actions(st.squeeze(0), k=10)
                a = schema_a.unsqueeze(0) if (schema_a is not None and conf > 0.3) else Normal(*pi(st)[:2]).sample()
                obs2, _, term, _, _ = env.step(a.squeeze(0).cpu().numpy())
                s = obs2
                if term:
                    gx, gy = env._goal_pos[0], env._goal_pos[1]
                    in_nw = gx < -1 and gy > 1
                    in_se = gx > 1 and gy < -1
                    if in_nw and side < 0: left_g += 1
                    elif in_se and side > 0: right_g += 1
                    else: wrong += 1
                    break
    return left_g, right_g, wrong


def run_trial(hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm, n_steps, env, is_bizonal=False, label=""):
    """Run practice steps in either maze or bizonal environment."""
    if is_bizonal:
        s = env.reset(seed=42)[0]
    else:
        s = env.reset(seed=42)[0]['state']
    
    goals, ep_s, ep_a, ep_r, ep_ns = 0, [], [], [], []
    ep_id = max(hc.ca3.episode_ids) + 1 if hc.ca3.episode_ids else 0
    da_boost, da_decay = 1.0, 0

    for step in range(n_steps):
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        schema_a, conf = hc.retrieve_actions(st.squeeze(0), k=10)
        a = schema_a.unsqueeze(0) if (schema_a is not None and conf > 0.3) else Normal(*pi(st)[:2]).sample()

        obs2, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
        if is_bizonal:
            s2 = obs2
        else:
            s2 = obs2['state']
        s2_t = torch.from_numpy(s2).float().to(DEVICE).unsqueeze(0)

        sp, rp = raw_fm(st, a)
        with torch.no_grad():
            Z = hc.ca3._get_Z(st.device)
            novelty = 1.0 - torch.softmax(hc.dg(st) @ Z.T * 5.0, dim=-1).max().item()
        total_rew = re + 0.1 * novelty
        dopamine_update(pi, opt_pi, opt_val, st, a, total_rew, s2_t, dopamine_boost=da_boost)
        F.mse_loss(sp, s2_t.detach()).backward()
        opt_raw_fm.step(); opt_raw_fm.zero_grad()
        hc.store(st.squeeze(0), a.squeeze(0), re, s2_t.squeeze(0), episode_id=ep_id)

        ep_s.append(st.squeeze(0).cpu()); ep_a.append(a.squeeze(0).cpu())
        ep_r.append(re); ep_ns.append(s2_t.squeeze(0).cpu())

        if da_decay > 0: da_decay -= 1; da_boost = 1.0 + 4.0 * (da_decay / 25.0)
        else: da_boost = 1.0

        if term:
            goals += 1; da_boost = 5.0; da_decay = 25
            if len(ep_s) > 1:
                for i in range(len(ep_s)):
                    hc.store(ep_s[i], ep_a[i], ep_r[i], ep_ns[i], episode_id=ep_id)

        if term or trunc:
            ep_id += 1
            if is_bizonal:
                s = env.reset(seed=None)[0]
            else:
                s = env.reset(seed=None)[0]['state']
            ep_s, ep_a, ep_r, ep_ns = [], [], [], []
        else:
            s = s2

        if step > 0 and step % 800 == 0 and len(hc) >= 50:
            all_idx = list(range(len(hc)))
            hs, ha, hr, hn, hg = hc.get_batch(all_idx)
            hs, ha, hn = hs.to(DEVICE), ha.to(DEVICE), hn.to(DEVICE)
            hr, hg = hr.to(DEVICE), hg.to(DEVICE)
            init_l = F.mse_loss(raw_fm(hs, ha)[0], hn).item()
            ep_ids = list(hc.ca3.episode_trajs.keys())
            for _ in range(500):
                eid = ep_ids[np.random.randint(len(ep_ids))]
                traj = hc.ca3.episode_trajs[eid]
                start = np.random.randint(0, max(1, len(traj) - 128))
                seg = traj[start:min(start + 128, len(traj))]
                ss = torch.stack([hc.ca3.states[i] for i in seg]).to(DEVICE)
                sa = torch.stack([hc.ca3.actions[i] for i in seg]).to(DEVICE)
                sn = torch.stack([hc.ca3.next_states[i] for i in seg]).to(DEVICE)
                sr = torch.tensor([hc.ca3.rewards[i] for i in seg]).to(DEVICE)
                sp_, rp_ = raw_fm(ss, sa)
                F.mse_loss(sp_, sn).backward()
                opt_raw_fm.step(); opt_raw_fm.zero_grad()

            hc.schema.update_from_ca3(hc, logger=logger)
            all_ep_ids = sorted(hc.ca3.episode_trajs.keys())
            if all_ep_ids:
                n_pp = max(1, 300 // len(all_ep_ids))
                bc_s, bc_a = [], []
                for eid in all_ep_ids:
                    traj = hc.ca3.episode_trajs[eid]
                    idx = np.random.choice(traj, min(n_pp, len(traj)), replace=False)
                    bc_s.extend([hc.ca3.states[i] for i in idx])
                    bc_a.extend([hc.ca3.actions[i] for i in idx])
                if bc_s:
                    bs = torch.stack(bc_s).to(DEVICE); ba = torch.stack(bc_a).to(DEVICE)
                    for _ in range(50):
                        F.mse_loss(pi(bs)[0], ba).backward()
                        opt_pi.step(); opt_pi.zero_grad()
            del hs, ha, hn, hr; gc.collect(); torch.cuda.empty_cache()
    return goals


def main():
    logger.info("═" * 70)
    logger.info("CROSS-ENVIRONMENT RETENTION TEST")
    logger.info("  Same agent. Same policy. Two different environments.")
    logger.info("  Tests: does learning one OVERWRITE the other?")
    logger.info("═" * 70)

    hc = Hippocampus(); pi = Policy().to(DEVICE); raw_fm = CerebellarModel().to(DEVICE)
    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)

    # ── Phase 1: Seed maze demos ──
    logger.info("\n📚 PHASE 1: Seed maze demos")
    from train_sr import run_demo
    demo = run_demo(NavArena(render_mode=None), n_trajs=25)
    for i in range(len(demo[0])):
        hc.store(demo[0][i], demo[1][i], demo[2][i], demo[3][i], episode_id=i // 56)
    ds = torch.stack(demo[0]).to(DEVICE); da = torch.stack(demo[1]).to(DEVICE)
    dn = torch.stack(demo[3]).to(DEVICE); dr = torch.tensor(demo[2], device=DEVICE)
    opt_r = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)
    for _ in range(200):
        F.mse_loss(raw_fm(ds, da)[0], dn).backward(); opt_r.step(); opt_r.zero_grad()
    opt_p = torch.optim.Adam(pi.parameters(), lr=1e-3)
    for _ in range(200):
        F.mse_loss(pi(ds)[0], da).backward(); opt_p.step(); opt_p.zero_grad()
    hc.schema.update_from_ca3(hc, logger=logger)
    logger.info(f"  → {len(hc)} patterns, {len(hc.schema)} schemas")

    # ── Phase 2: Test maze retention ──
    logger.info("\n🧪 PHASE 2: Maze retention (baseline)")
    env = NavArena(render_mode=None)
    maze_before = test_maze(env, hc, pi, n_eps=10)
    env.close()
    for k, v in maze_before.items():
        logger.info(f"  {k}: {v}/10")

    # ── Phase 3: Learn Bizonal ──
    logger.info("\n📚 PHASE 3: Learn Bizonal zone rules")
    # Seed bizonal demos
    benv = BizonalArena(render_mode=None)
    for side in [-1, 1]:
        for t in range(10):
            obs, _ = benv.reset(seed=t + (0 if side < 0 else 100))
            s = obs
            for _ in range(200):
                g = benv._goal_pos[:2]; p = s[:2]
                d_vec = g - p; dist = np.linalg.norm(d_vec)
                a = np.clip(d_vec / dist if dist > .2 else d_vec * .5, -1, 1).astype(np.float32)
                obs2, r, term, _, _ = benv.step(a)
                s2 = obs2
                hc.store(torch.tensor(s), torch.tensor(a), r, torch.tensor(s2), episode_id=1000 + t)
                s = s2
                if term: break
    benv.close()
    logger.info(f"  Seeded: {len(hc)} patterns")

    # Practice Bizonal
    benv = BizonalArena(render_mode=None)
    biz_goals = run_trial(hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm, 2000, benv, is_bizonal=True, label="Bizonal")
    benv.close()
    logger.info(f"  Practice: {biz_goals} goals")

    # ── Phase 4: Test Bizonal retention ──
    logger.info("\n🧪 PHASE 4: Bizonal retention")
    benv = BizonalArena(render_mode=None)
    l, r, w = test_bizonal(benv, hc, pi, n_eps=15)
    benv.close()
    logger.info(f"  LEFT→NW: {l}/15  RIGHT→SE: {r}/15  wrong_zone: {w}")

    # ── Phase 5: Test maze retention AGAIN ──
    logger.info("\n🧪 PHASE 5: Maze retention (after Bizonal)")
    env = NavArena(render_mode=None)
    maze_after = test_maze(env, hc, pi, n_eps=10)
    env.close()
    for k, v in maze_after.items():
        logger.info(f"  {k}: {v}/10")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("CROSS-ENVIRONMENT RETENTION RESULTS")
    print("=" * 70)
    print(f"\nMaze retention (BEFORE Bizonal):")
    for k, v in maze_before.items():
        print(f"  {k:15s}: {v}/10")
    print(f"\nBizonal retention:")
    print(f"  LEFT→NW       : {l}/{15}  (wrong_zone: {w})")
    print(f"  RIGHT→SE      : {r}/{15}  (wrong_zone: {w})")
    print(f"\nMaze retention (AFTER Bizonal):")
    for k, v in maze_after.items():
        change = v - maze_before[k]
        print(f"  {k:15s}: {v}/10 {'+' if change >= 0 else ''}{change}")
    retained = sum(1 for k in maze_before if maze_after[k] >= maze_before[k])
    print(f"\nRetention: {retained}/5 maze phases unchanged or improved")
    print(f"Final: {len(hc)} patterns, {len(hc.schema)} schemas")
    vr = torch.cuda.memory_allocated() / 1e6
    print(f"VRAM: {vr:.0f}MB")


if __name__ == '__main__':
    main()
