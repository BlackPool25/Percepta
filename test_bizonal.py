"""Multi-Rule Arena: test context-dependent rule learning.
LEFT start  → goal in NW quadrant (x < -1, y > 1)
RIGHT start → goal in SE quadrant (x > 1, y < -1)

Agent must INFER the rule from its start position and apply it.
"""

import torch, torch.nn.functional as F, gc, time, logging, numpy as np
from torch.distributions import Normal
from pathlib import Path
from env_bizonal import BizonalArena
from train_sr import (Hippocampus, Policy, CerebellarModel,
                      dopamine_update, DEVICE)

OUT = Path('results/bizonal')
OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
    handlers=[logging.FileHandler(OUT / 'bizonal.log', mode='w'), logging.StreamHandler()])
logger = logging.getLogger('bizonal')

S = 4  # state dim for bizonal arena (pos + vel only)


def seed_agent(hc, pi, raw_fm):
    """Seed with oracle demos showing the zone rule."""
    env = BizonalArena(render_mode=None)
    all_s, all_a, all_r, all_ns = [], [], [], []

    for side, label in [(-1, "LEFT"), (1, "RIGHT")]:
        for traj in range(10):
            env.reset(seed=traj + (0 if side < 0 else 100))
            s = env._get_obs()
            for _ in range(200):
                g = env._goal_pos[:2]; p = s[:2]
                d_vec = g - p; dist = np.linalg.norm(d_vec)
                a = np.clip(d_vec / dist if dist > .2 else d_vec * .5, -1, 1).astype(np.float32)
                obs2, r, term, _, _ = env.step(a)
                s2 = obs2
                all_s.append(torch.tensor(s)); all_a.append(torch.tensor(a))
                all_r.append(float(r)); all_ns.append(torch.tensor(s2))
                s = s2
                if term: break
        logger.info(f"  {label}: {sum(1 for _s in all_s if _s[0].item() * side > 0)} last-side trajectories")

    env.close()
    logger.info(f"Total: {len(all_s)} seed transitions")

    for i in range(len(all_s)):
        hc.store(all_s[i], all_a[i], all_r[i], all_ns[i], episode_id=i // 50)

    ds = torch.stack(all_s).to(DEVICE)
    da = torch.stack(all_a).to(DEVICE)
    dn = torch.stack(all_ns).to(DEVICE)
    dr = torch.tensor(all_r, device=DEVICE)
    
    opt_r = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)
    for _ in range(500):
        sp, rp = raw_fm(ds, da)
        F.mse_loss(sp, dn).backward()
        opt_r.step(); opt_r.zero_grad()
    
    opt_p = torch.optim.Adam(pi.parameters(), lr=1e-3)
    for _ in range(500):
        m, _, _ = pi(ds)
        F.mse_loss(m, da).backward()
        opt_p.step(); opt_p.zero_grad()
    
    hc.schema.update_from_ca3(hc, logger=logger)
    logger.info(f"Seeded: {len(hc)} patterns, {len(hc.schema)} schemas")


def practice(env, hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm, n_steps=2000):
    """Practice in mixed zones. Agent must learn zone-specific rules."""
    s = env.reset(seed=42)[0]
    goals, ep_s, ep_a, ep_r, ep_ns = 0, [], [], [], []
    ep_id = 0
    da_boost, da_decay = 1.0, 0

    for step in range(n_steps):
        st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        schema_a, conf = hc.retrieve_actions(st.squeeze(0), k=10)
        a = schema_a.unsqueeze(0) if (schema_a is not None and conf > 0.3) else Normal(*pi(st)[:2]).sample()

        obs2, re, term, trunc, _ = env.step(a.squeeze(0).cpu().numpy())
        s2 = obs2
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
            ep_id += 1; s = env.reset(seed=None)[0]
            ep_s, ep_a, ep_r, ep_ns = [], [], [], []
        else:
            s = s2

        # Sleep
        if step > 0 and step % 400 == 0 and len(hc) >= 50:
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
            final_l = F.mse_loss(raw_fm(hs, ha)[0], hn).item()
            logger.info(f"  Sleep: RawFM {init_l:.4f}→{final_l:.4f}")

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

        if step % 400 == 0:
            vr = torch.cuda.memory_allocated() / 1e6
            logger.info(f"  step={step:4d} goals={goals} hc={len(hc)} vr={vr:.0f}MB")

    return goals


def test_retention(env, hc, pi, n_eps=20):
    """Test: does the agent apply the CORRECT zone rule?"""
    results = {}
    for side, label in [(-1, "LEFT→NW"), (1, "RIGHT→SE")]:
        goals, nw_goals, se_goals = 0, 0, 0
        for ep in range(n_eps):
            obs, _ = env.reset(seed=ep * 10 + 42)
            start_x = env.data.xpos[env._agent_body_id][0]
            if start_x * side < 0:
                continue
            s = obs
            for _ in range(500):
                st = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
                schema_a, conf = hc.retrieve_actions(st.squeeze(0), k=10)
                a = schema_a.unsqueeze(0) if (schema_a is not None and conf > 0.3) else Normal(*pi(st)[:2]).sample()
                obs2, _, term, _, _ = env.step(a.squeeze(0).cpu().numpy())
                s = obs2
                if term:
                    goals += 1
                    gx, gy = env._goal_pos[0], env._goal_pos[1]
                    if gx < -1 and gy > 1: nw_goals += 1
                    if gx > 1 and gy < -1: se_goals += 1
                    break
        results[label] = (goals, n_eps, nw_goals, se_goals)
        correct_zone = nw_goals if side < 0 else se_goals
        logger.info(f"  {label}: {goals}/{n_eps} = {goals/n_eps:.0%} (correct_zone={correct_zone}, wrong_zone={se_goals if side < 0 else nw_goals})")
    return results


def main():
    logger.info("═" * 70)
    logger.info("MULTI-RULE ARENA: Context-Dependent Goal Learning")
    logger.info("  LEFT start  → goal in NW quadrant")
    logger.info("  RIGHT start → goal in SE quadrant")
    logger.info("═" * 70)

    # Init with bizonal state dimension
    # Init with standard state dimension (matches original maze env for shared use)
    hc = Hippocampus(state_dim=10)
    pi = Policy().to(DEVICE)
    raw_fm = CerebellarModel().to(DEVICE)

    opt_pi = torch.optim.Adam([
        {'params': pi.shared.parameters(), 'lr': 1e-3},
        {'params': pi.mean.parameters(), 'lr': 1e-3},
        {'params': pi.log_std, 'lr': 1e-3},
    ])
    opt_val = torch.optim.Adam(pi.value.parameters(), lr=1e-3)
    opt_raw_fm = torch.optim.Adam(raw_fm.parameters(), lr=1e-3)

    # Phase 1: Seed
    logger.info("\n📚 SEEDING: Oracle demos for both zones")
    seed_agent(hc, pi, raw_fm)

    # Phase 2: Practice
    logger.info("\n✏️  PRACTICE: Mixed-zone navigation")
    env = BizonalArena(render_mode=None)
    goals = practice(env, hc, pi, raw_fm, opt_pi, opt_val, opt_raw_fm, 4000)
    logger.info(f"  → {goals} goals in practice")
    env.close()

    # Phase 3: Retention test
    logger.info("\n🧪 RETENTION TEST: Novel starts in both zones")
    env = BizonalArena(render_mode=None)
    results = test_retention(env, hc, pi, n_eps=20)
    env.close()

    print("\n" + "=" * 60)
    print("ZONE-RULE LEARNING RESULTS")
    print("=" * 60)
    for label, (g, n, nw, se) in results.items():
        print(f"  {label:15s}: {g}/{n} = {g/n:.0%}")
        correct = nw if "LEFT" in label else se
        wrong = se if "LEFT" in label else nw
        print(f"  {'':15s}  correct_zone={correct}, wrong_zone={wrong}")
    print(f"\n📊 Final: {len(hc)} patterns, {len(hc.schema)} schemas")
    vr = torch.cuda.memory_allocated() / 1e6
    logger.info(f"   VRAM: {vr:.0f}MB")


if __name__ == '__main__':
    main()
