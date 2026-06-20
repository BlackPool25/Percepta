"""Full Percepta agent combining fast/slow encoder + RSSM + actor-critic.

Within-episode (tick):
  1. PerceptaModel encodes obs → features (fast paths adapt)
  2. RSSM recurrent_step → h (deterministic latent)
  3. RSSM infer_posterior(h, features) → z (stochastic latent)
  4. RSSM ensemble_curiosity(h) → curiosity reward
  5. Actor(h, z) → action
  6. Fast paths update stability

Between-episode (sleep):
  1. Train RSSM on collected sequences (reconstruction + KL + reward)
  2. Imagine sequences from RSSM using policy
  3. Train actor-critic on imagined trajectories (Dreamer-style)
  4. Gate: consolidate important experiences
  5. Decay fast weights toward init
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import PerceptaModel, FastDecayPath
from world_model import RSSM


class Actor(nn.Module):
    """Stochastic policy: (h, z) → action distribution."""
    def __init__(self, deter_dim: int = 64, stoch_dim: int = 32,
                 action_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, h: torch.Tensor, z: torch.Tensor):
        x = torch.cat([h, z], dim=-1)
        x = self.net(x)
        mean = torch.tanh(self.mean(x))
        std = F.softplus(self.log_std) + 0.01
        return mean, std

    def sample(self, h: torch.Tensor, z: torch.Tensor,
               deterministic: bool = False) -> torch.Tensor:
        mean, std = self.forward(h, z)
        if deterministic:
            return mean
        return torch.tanh(mean + torch.randn_like(std) * std)

    def log_prob(self, h: torch.Tensor, z: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(h, z)
        # TanhNormal log-prob (accounting for tanh squashing)
        mean_t = torch.arctanh(action.clamp(-0.999, 0.999))
        logp = -0.5 * (torch.log(2 * torch.pi * std ** 2) +
                       ((mean_t - mean) / std) ** 2).sum(dim=-1)
        # Tanh correction
        logp -= torch.log(1 - action ** 2 + 1e-8).sum(dim=-1)
        return logp


class Critic(nn.Module):
    """Value function: (h, z) → scalar state value."""
    def __init__(self, deter_dim: int = 64, stoch_dim: int = 32,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class PerceptaAgent(nn.Module):
    """Full agent: PerceptaModel + RSSM + Actor + Critic."""

    def __init__(self, obs_dim: int = 33, task_dim: int = 3,
                 action_dim: int = 3, feat_dim: int = 64,
                 deter_dim: int = 64, stoch_dim: int = 32,
                 fast_gen_dim: int = 64, fast_spec_dim: int = 32,
                 fast_resid_dim: int = 16, n_ensemble: int = 3,
                 curiosity_scale: float = 0.1, task_scale: float = 1.0,
                 device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.curiosity_scale = curiosity_scale
        self.task_scale = task_scale
        self.feat_dim = feat_dim

        self.percepta = PerceptaModel(
            obs_dim=obs_dim, task_dim=task_dim, feat_dim=feat_dim,
            fast_gen_dim=fast_gen_dim, fast_spec_dim=fast_spec_dim,
            fast_resid_dim=fast_resid_dim,
        )
        self.rssm = RSSM(
            obs_dim=obs_dim, action_dim=action_dim,
            feat_dim=feat_dim, deter_dim=deter_dim,
            stoch_dim=stoch_dim, n_ensemble=n_ensemble,
        )
        self.actor = Actor(deter_dim, stoch_dim, action_dim)
        self.critic = Critic(deter_dim, stoch_dim)

        self.to(self.device)

        # Persistent optimizers (created once, reused across episodes)
        self.ae_opt = torch.optim.Adam(self.percepta.parameters(), lr=1e-3)
        self.rssm_opt = torch.optim.Adam(self.rssm.parameters(), lr=1e-3)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    @torch.no_grad()
    def tick(self, obs: np.ndarray, task: np.ndarray,
             h: torch.Tensor | None = None,
             z: torch.Tensor | None = None,
             action_prev: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Process one timestep: update state given new observation.

        Input: previous internal state (h, z) and the action that was just taken.
        Updates h via recurrence, then infers z from the new observation.

        Returns: (h, z, curiosity, feat) for the current timestep.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        task_t = torch.as_tensor(task, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
        feat = self.percepta(obs_t, task_t)

        if h is None or z is None:
            h = self.rssm.encode_obs(feat)
        else:
            h = self.rssm.recurrent_step(h, z, action_prev)

        curiosity, _ = self.rssm.ensemble_curiosity(h)
        z, _, _ = self.rssm.infer_posterior(h, feat)

        h = h.detach()
        z = z.detach()
        return h, z, curiosity.item(), feat

    @torch.no_grad()
    def select_action(self, h: torch.Tensor, z: torch.Tensor,
                      deterministic: bool = False) -> np.ndarray:
        action = self.actor.sample(h, z, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def mpc_plan(self, h: torch.Tensor, z: torch.Tensor,
                 n_candidates: int = 50, horizon: int = 10) -> np.ndarray:
        """Model Predictive Control: plan best action via RSSM simulation.

        Simulates n_candidates random action sequences of length horizon
        in the RSSM, scores them by goal proximity (goal head predicts
        -distance), and returns the first action from the best sequence.

        This is the agent's 'deliberate reasoning' — thinking before acting.
        """
        K, H = n_candidates, horizon
        device = h.device

        # Generate random action sequences
        # Include the actor's suggested action as one candidate
        actor_action = self.actor.sample(h, z, deterministic=False)
        actions = torch.randn(K - 1, H, self.action_dim, device=device).tanh()

        # Insert actor's action as first action of first candidate
        # (this biases MPC toward the learned policy)
        all_actions = torch.zeros(K, H, self.action_dim, device=device)
        all_actions[0, 0] = actor_action
        all_actions[1:] = actions

        h_batch = h.expand(K, -1)
        z_batch = z.expand(K, -1)

        total_prox = torch.zeros(K, device=device)

        for t in range(H):
            h_batch = self.rssm.recurrent_step(h_batch, z_batch, all_actions[:, t])
            z_batch, _, _ = self.rssm.predict_prior(h_batch)
            prox = self.rssm.predict_goal_proximity(h_batch, z_batch)
            total_prox += prox

        best_idx = total_prox.argmax()
        return all_actions[best_idx, 0].cpu().numpy()

    @torch.no_grad()
    def compute_curiosity(self, obs: np.ndarray, task: np.ndarray,
                          h: torch.Tensor) -> float:
        """Compute curiosity reward for this timestep."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        task_t = torch.as_tensor(task, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            feat = self.percepta(obs_t, task_t)
            if h is None or h.numel() == 0:
                h = self.rssm.encode_obs(feat)
            curiosity, _ = self.rssm.ensemble_curiosity(h)
        return self.curiosity_scale * curiosity.item()

    def update_fast_stabilities(self):
        self.percepta.update_fast_stabilities()

    def decay_fast_weights(self):
        self.percepta.decay_fast_weights()

    def get_confidences(self) -> float:
        return self.percepta.get_confidence()

    def get_slow_params(self):
        return self.percepta.get_slow_params()

    def get_fast_params(self):
        return self.percepta.get_fast_params()

    def train_rssm(self, sequence: dict, lr: float = 1e-3,
                   kl_scale: float = 0.1) -> dict:
        """Train RSSM on a sequence of experience.

        sequence must contain:
          feats: (T, B, feat_dim) — already encoded by PerceptaModel
          actions: (T, B, action_dim)
          rewards: (T, B)
        """
        opt = torch.optim.Adam(self.rssm.parameters(), lr=lr)

        feats = torch.as_tensor(sequence['feats'], device=self.device)
        actions = torch.as_tensor(sequence['actions'], device=self.device)
        rewards = torch.as_tensor(sequence['rewards'], device=self.device)

        opt.zero_grad()
        out = self.rssm.forward_sequence(feats, actions, rewards)
        loss = (out['recon_loss'] + out['reward_loss']
                + kl_scale * out['kl_loss'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 10.0)
        opt.step()

        return {
            'rssm_loss': loss.item(),
            'recon_loss': out['recon_loss'].item(),
            'kl_loss': out['kl_loss'].item(),
            'reward_loss': out['reward_loss'].item(),
        }

    def train_policy(self, h_start: torch.Tensor, z_start: torch.Tensor,
                     horizon: int = 50, lr: float = 3e-4,
                     discount: float = 0.99, lambda_: float = 0.95,
                     n_imaginations: int = 1) -> dict:
        """Train actor and critic via imagined rollouts (Dreamer-style).

        Args:
          h_start: (B, deter_dim) starting deterministic latents
          z_start: (B, stoch_dim) starting stochastic latents
          horizon: imagination horizon
          lr: learning rate for actor and critic
          discount: discount factor
          lambda_: TD(lambda) parameter

        Returns dict of losses.
        """
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        for _ in range(n_imaginations):
            # Imagine trajectory
            imag = self.rssm.imagine_sequence(
                h_start, z_start,
                lambda h, z: self.actor.sample(h, z, deterministic=False),
                horizon,
            )
            h_imag = imag['h']       # (horizon, B, deter_dim)
            z_imag = imag['z']       # (horizon, B, stoch_dim)
            rew_imag = imag['reward']  # (horizon, B)

            # Compute values
            with torch.no_grad():
                values = self.critic(h_imag, z_imag)  # (horizon, B)

            # Compute lambda returns
            returns = self._compute_lambda_returns(
                rew_imag, values, discount, lambda_
            )

            # Actor loss: negative advantage
            with torch.no_grad():
                advantage = returns - values

            # Need actions for log_prob (re-sample from actor at imagined states)
            # Re-sample to compute log_prob
            actions_imag = imag['action']
            log_probs = self.actor.log_prob(h_imag, z_imag, actions_imag)
            actor_loss = -(log_probs * advantage.detach()).mean()

            # Critic loss: MSE between value and return
            critic_loss = F.mse_loss(values, returns)

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            critic_opt.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
        }

    @staticmethod
    def _compute_lambda_returns(rewards: torch.Tensor,
                                 values: torch.Tensor,
                                 discount: float = 0.99,
                                 lambda_: float = 0.95) -> torch.Tensor:
        """TD(lambda) returns from imagined trajectory."""
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        g = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                g = rewards[t] + discount * values[t]
            else:
                g = rewards[t] + discount * (
                    (1 - lambda_) * values[t] + lambda_ * g
                )
            returns[t] = g
        return returns
