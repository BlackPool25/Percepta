"""Episodic stream training loop with 3-phase exploration + JEPA + autoencoding.

Phases:
  1. Count-based exploration (ep 0-200): random noise + position-count bonus
  2. Curiosity-driven exploration (ep 200-500): ensemble disagreement bonus
  3. Task exploitation (ep 500+): curiosity decays, task reward dominates

Training during sleep:
  - PerceptaModel: self-supervised autoencoding (stable, independent)
  - RSSM: JEPA objective (predict stop-gradient features)
  - Actor-Critic: Dreamer-style imagination training

Speed: subsequence sampling for RSSM (O(subseq_len) not O(T))
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from env import MuJoCoPlayground
from agent import PerceptaAgent


def get_phase(episode: int) -> str:
    """Return current training phase based on episode number."""
    if episode < 200:
        return 'explore_count'
    elif episode < 500:
        return 'explore_curiosity'
    else:
        return 'exploit'


def get_exploration_bonus(obs: np.ndarray, visit_counts: defaultdict,
                          episode: int) -> float:
    """Count-based exploration bonus based on agent position bins."""
    phase = get_phase(episode)
    if phase == 'explore_count':
        x, y = obs[0], obs[1]
        bin_x = int(np.clip((x + 5) * 2, 0, 19))
        bin_y = int(np.clip((y + 5) * 2, 0, 19))
        key = (bin_x, bin_y)
        count = visit_counts.get(key, 0)
        visit_counts[key] = count + 1
        return 0.02 / np.sqrt(count + 1.0)  # modest — task reward should drive behavior
    return 0.0


def get_curiosity_scale(episode: int) -> float:
    """Decaying curiosity weight across phases. Task reward should dominate."""
    if episode < 200:
        return 0.0
    elif episode < 400:
        frac = (episode - 200) / 200.0
        return 0.15 * frac  # max 0.15 — modest, task reward dominates
    else:
        frac = min(1.0, (episode - 400) / 200.0)
        return max(0.0, 0.15 * (1.0 - frac))  # decay to zero


def heuristic_action(obs: np.ndarray, task: np.ndarray, noise: float = 0.2) -> np.ndarray:
    """Prior knowledge: position behind object, push toward goal."""
    target_idx = np.argmax(task)
    obj_pos = obs[6 + target_idx * 6: 9 + target_idx * 6]
    agent_pos = obs[0:3]
    goal_pos = obs[24:27]

    # Push direction: from object away from goal (we push object toward goal)
    push_dir = obj_pos[:2] - goal_pos[:2]
    push_dist = np.linalg.norm(push_dir)
    if push_dist > 0.01:
        push_dir_unit = push_dir / push_dist
    else:
        push_dir_unit = np.zeros(2)

    # Desired position: behind the object relative to goal
    target_pos = obj_pos[:2] + push_dir_unit * 0.6

    # Move toward target position
    dir_to_target = target_pos - agent_pos[:2]
    dist_to_target = np.linalg.norm(dir_to_target)

    if dist_to_target > 0.3:
        # Approach pushing position
        action_2d = dir_to_target / (dist_to_target + 0.01)
        action_2d *= min(1.0, dist_to_target * 2.0)
    else:
        # In position: push object toward goal
        goal_dir = goal_pos[:2] - obj_pos[:2]
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 0.01:
            action_2d = goal_dir / goal_dist
        else:
            action_2d = np.zeros(2)
        # Slow, controlled push
        action_2d *= min(1.0, goal_dist)
        action_2d *= 0.3

    action = np.array([action_2d[0], action_2d[1], -0.3])
    action += np.random.randn(3) * noise
    return np.clip(action, -1, 1)


def run_episode(env, agent, max_steps, episode, visit_counts,
                training=True, curiosity_scale=0.0, use_mpc=False,
                heuristic_prob=0.0):
    """Run one episode. heuristic_prob = probability of using heuristic action."""
    obs, info = env.reset()
    task = np.zeros(3, dtype=np.float32)
    task[info['target_object']] = 1.0

    h, z = None, None
    action = np.zeros(agent.action_dim, dtype=np.float32)

    obs_list, task_list, action_list = [], [], []
    reward_list, bonus_list = [], []
    feat_list = []
    h_list, z_list = [], []
    prox_list = []

    episode_reward = 0.0

    for step in range(max_steps):
        action_t = torch.as_tensor(action, dtype=torch.float32,
                                   device=agent.device).unsqueeze(0)
        h, z, curiosity, feat = agent.tick(obs, task, h, z, action_t)

        obs_list.append(obs)
        task_list.append(task)
        feat_list.append(feat.squeeze(0).cpu().numpy())
        h_list.append(h.squeeze(0).cpu().numpy())
        z_list.append(z.squeeze(0).cpu().numpy())

        # Goal proximity (for MPC training)
        with torch.no_grad():
            prox_list.append(agent.rssm.predict_goal_proximity(h, z).item())

        # Exploration bonus
        count_bonus = get_exploration_bonus(obs, visit_counts, episode)
        curiosity_bonus = curiosity_scale * curiosity
        exploration_bonus = count_bonus + curiosity_bonus
        bonus_list.append(exploration_bonus)

        if use_mpc and training:
            action = agent.mpc_plan(h, z, n_candidates=50, horizon=10)
        elif training and np.random.random() < heuristic_prob:
            action = heuristic_action(obs, task, noise=max(0.05, heuristic_prob))
        else:
            action = agent.select_action(h, z, deterministic=not training)
        action_list.append(action)

        next_obs, env_reward, terminated, truncated, info = env.step(action)
        next_task = np.zeros(3, dtype=np.float32)
        next_task[info['target_object']] = 1.0

        total_reward = env_reward + exploration_bonus
        reward_list.append(total_reward)
        episode_reward += total_reward

        if training:
            agent.update_fast_stabilities()

        obs, task = next_obs, next_task
        if terminated or truncated:
            break

    return {
        'obs': np.stack(obs_list),
        'task': np.stack(task_list),
        'feat': np.stack(feat_list),
        'action': np.stack(action_list),
        'reward': np.array(reward_list, dtype=np.float32),
        'h': np.stack(h_list),
        'z': np.stack(z_list),
        'episode_reward': episode_reward,
        'episode_length': len(reward_list),
        'exploration_bonus': np.sum(bonus_list),
        'goal_prox': np.array(prox_list, dtype=np.float32),
    }


def sleep_phase(agent, episode_data, ae_buffer=None,
                ae_lr=1e-3, rssm_lr=1e-3, policy_lr=3e-4,
                kl_scale=0.1, imagination_horizon=50,
                n_ae_steps=10, n_rssm_steps=10, n_policy_steps=10,
                subseq_len=16,
                discount=0.99, lambda_=0.95):
    """Between-episode consolidation (optimized)."""
    T = episode_data['obs'].shape[0]
    device = agent.device

    # Move ALL data to GPU once, reuse across steps
    obs_all = torch.as_tensor(episode_data['obs'], dtype=torch.float32, device=device)
    task_all = torch.as_tensor(episode_data['task'], dtype=torch.float32, device=device)
    feats_all = torch.as_tensor(episode_data['feat'], dtype=torch.float32, device=device)
    actions_all = torch.as_tensor(episode_data['action'], dtype=torch.float32, device=device)
    rewards_all = torch.as_tensor(episode_data['reward'], dtype=torch.float32, device=device)

    ae_losses, rssm_losses, actor_losses = [], [], []

    # Phase 1: PerceptaModel autoencoding
    # Sample from replay buffer if available (prevents forgetting old observations)
    ae_source = ae_buffer if ae_buffer and len(ae_buffer) >= subseq_len else episode_data
    if isinstance(ae_source, list):  # buffer mode: sample random observations
        indices = np.random.randint(0, len(ae_source), size=subseq_len)
        o = torch.stack([ae_source[i][0] for i in indices])
        t = torch.stack([ae_source[i][1] for i in indices])
    else:  # single episode mode: sample subsequence
        start = np.random.randint(0, max(1, T - subseq_len))
        sl = min(subseq_len, T - start)
        o = obs_all[start:start + sl]
        t = task_all[start:start + sl]

    for _ in range(n_ae_steps):
        # Re-sample each step if using buffer
        if isinstance(ae_source, list) and len(ae_source) >= subseq_len:
            indices = np.random.randint(0, len(ae_source), size=subseq_len)
            o = torch.stack([ae_source[i][0] for i in indices]).to(device)
            t = torch.stack([ae_source[i][1] for i in indices]).to(device)
        elif not isinstance(ae_source, list):
            start = np.random.randint(0, max(1, T - subseq_len))
            sl = min(subseq_len, T - start)
            o = obs_all[start:start + sl]
            t = task_all[start:start + sl]

        agent.ae_opt.zero_grad()
        loss = agent.percepta.ae_loss(o, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.percepta.parameters(), 10.0)
        agent.ae_opt.step()
        ae_losses.append(loss.item())

    # Phase 2: RSSM JEPA training (reuse optimizer)
    for _ in range(n_rssm_steps):
        start = np.random.randint(0, max(1, T - subseq_len))
        sl = min(subseq_len, T - start)
        f = feats_all[start:start + sl].unsqueeze(1)
        a = actions_all[start:start + sl].unsqueeze(1)
        r = rewards_all[start:start + sl].unsqueeze(1)

        agent.rssm_opt.zero_grad()
        out = agent.rssm.forward_sequence(f, a, r)
        loss = out['recon_loss'] + out['reward_loss'] + kl_scale * out['kl_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.rssm.parameters(), 10.0)
        agent.rssm_opt.step()
        rssm_losses.append(loss.item())

    # Phase 3: Actor-critic imagination training (reuse optimizers)
    if n_policy_steps > 0 and episode_data['h'].size > 0:
        h_start = torch.as_tensor(episode_data['h'][-1:], dtype=torch.float32, device=device)
        z_start = torch.as_tensor(episode_data['z'][-1:], dtype=torch.float32, device=device)

        for _ in range(n_policy_steps):
            with torch.no_grad():
                imag = agent.rssm.imagine_sequence(
                    h_start, z_start,
                    lambda h, z: agent.actor.sample(h, z, deterministic=False),
                    imagination_horizon,
                )
                h_i, z_i, a_i, r_i = imag['h'], imag['z'], imag['action'], imag['reward']
                v = agent.critic(h_i, z_i)
                ret = agent._compute_lambda_returns(r_i, v, discount, lambda_)
                adv = ret - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            lp = agent.actor.log_prob(h_i, z_i, a_i)
            al = -(lp * adv).mean()
            v2 = agent.critic(h_i, z_i)
            cl = F.mse_loss(v2, ret)

            agent.actor_opt.zero_grad(); al.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
            agent.actor_opt.step()

            agent.critic_opt.zero_grad(); cl.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 10.0)
            agent.critic_opt.step()

            actor_losses.append(al.item())

    # Phase 4: Fast weight decay
    agent.decay_fast_weights()

    return {
        'ae_loss': np.mean(ae_losses[-5:]),
        'rssm_loss': np.mean(rssm_losses[-5:]),
        'actor_loss': np.mean(actor_losses[-5:]) if actor_losses else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ae-lr', type=float, default=1e-3)
    parser.add_argument('--rssm-lr', type=float, default=1e-3)
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = MuJoCoPlayground(render_mode=None, max_steps=args.max_steps,
                           force_scale=50.0)
    agent = PerceptaAgent(device=device)
    visit_counts = defaultdict(int)

    # Replay buffer for autoencoding (prevents AE loss spikes)
    ae_buffer = []  # list of (obs, task) numpy arrays
    AE_BUF_MAX = 50000  # keep ~250 episodes of observations (prevents forgetting)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f'Device: {device} | Params: {total_params:,}')
    print(f'Phases: count-based (0-200) → curiosity (200-400) → exploit (400+)')

    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        phase = get_phase(ep)
        c_scale = get_curiosity_scale(ep)

        use_mpc = (ep > 20) and (ep < 100)  # MPC planning phase
        use_heuristic = ep <= 50  # prior knowledge bootstrap
        h_prob = max(0.0, 1.0 - ep / 50.0) if use_heuristic else 0.0

        data = run_episode(env, agent, args.max_steps, ep, visit_counts,
                          training=True, curiosity_scale=c_scale,
                          use_mpc=use_mpc, heuristic_prob=h_prob)

        # Add to AE replay buffer (keep recent observations)
        for t in range(min(len(data['obs']), 200)):
            obs_t = torch.as_tensor(data['obs'][t])
            task_t = torch.as_tensor(data['task'][t])
            ae_buffer.append((obs_t, task_t))
        if len(ae_buffer) > AE_BUF_MAX:
            ae_buffer = ae_buffer[-AE_BUF_MAX:]

        # Compute goal distances from episode data for goal head training
        goal_pos = np.array([3.0, 3.0, 0.0])
        goal_dists = []
        for t in range(len(data['obs'])):
            target_idx = np.argmax(data['task'][t])
            obj_pos = data['obs'][t, 6 + target_idx*6 : 9 + target_idx*6]
            dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
            goal_dists.append(dist)
        goal_dists_t = torch.as_tensor(goal_dists, dtype=torch.float32, device=device)

        losses = sleep_phase(
            agent, data, ae_buffer=ae_buffer,
            n_ae_steps=15, n_rssm_steps=15,
            n_policy_steps=10 if phase != 'explore_count' else 0,
        )

        # Train goal head separately (for MPC planning)
        feats = torch.as_tensor(data['feat'], dtype=torch.float32, device=device)
        for _ in range(10):
            start = np.random.randint(0, max(1, len(feats) - 16))
            f = feats[start:start + 16].unsqueeze(1)
            gd = goal_dists_t[start:start + 16].unsqueeze(1)
            agent.rssm_opt.zero_grad()
            out = agent.rssm.forward_sequence(f, f[:16].unsqueeze(1) * 0,
                                           f[:16].unsqueeze(1) * 0,
                                           goal_dists=gd)
            if out['goal_loss'] > 0:
                out['goal_loss'].backward()
                torch.nn.utils.clip_grad_norm_(agent.rssm.parameters(), 10.0)
                agent.rssm_opt.step()

        # Policy imitation: train actor to match MPC actions
        if use_mpc and 'goal_prox' in data:
            h_all = torch.as_tensor(data['h'], dtype=torch.float32, device=device)
            z_all = torch.as_tensor(data['z'], dtype=torch.float32, device=device)
            mpc_actions = torch.as_tensor(data['action'], dtype=torch.float32, device=device)
            for _ in range(10):
                idx = np.random.randint(0, len(h_all))
                h_s = h_all[idx:idx+1]
                z_s = z_all[idx:idx+1]
                a_mpc = mpc_actions[idx:idx+1]
                a_pred = agent.actor.sample(h_s, z_s, deterministic=False)
                # Behavioral cloning loss: MSE between actor output and MPC action
                bc_loss = F.mse_loss(a_pred, a_mpc)
                agent.actor_opt.zero_grad()
                bc_loss.backward()
                agent.actor_opt.step()

        if ep % 50 == 0 or ep == 1:
            avg_r = np.mean([data['episode_reward']])
            avg_bonus = np.mean([data['exploration_bonus']])
            unique_bins = len(visit_counts)
            elapsed = time.time() - start_time
            print(
                f'Ep {ep:4d} | {phase:15s} | '
                f'R={avg_r:6.1f} | Bonus={avg_bonus:5.1f} | '
                f'Bins={unique_bins:4d} | '
                f'AE={losses["ae_loss"]:.2f} | '
                f'RSSM={losses["rssm_loss"]:.2f} | '
                f'Act={losses["actor_loss"]:.3f} | '
                f'T={elapsed:.0f}s'
            )

    total_time = time.time() - start_time
    print(f'\n{args.episodes} episodes in {total_time:.0f}s ({total_time/args.episodes:.2f}s/ep)')

    # Final evaluation
    print('\nFinal evaluation...')
    eval_rewards = []
    for _ in range(10):
        d = run_episode(env, agent, args.max_steps, args.episodes,
                       visit_counts, training=False, curiosity_scale=0.0)
        eval_rewards.append(d['episode_reward'])
    print(f'  R={np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}')

    env.close()


if __name__ == '__main__':
    main()
