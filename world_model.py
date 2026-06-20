"""RSSM — Recurrent State-Space Model (Dreamer-style).

Flow:
  t=0: encode_obs(feat_0) → h_0
       infer_posterior(h_0, feat_0) → z_0_post  (informed by obs)
       predict_prior(h_0) → z_0_prior           (predicted from h alone)
       decode(h_0, z_0_post) → feat_0_recon     (reconstruct current)
       
  t=1..T-1: recurrent_step(h_{t-1}, z_{t-1}, action_{t-1}) → h_t
            infer_posterior(h_t, feat_t) → z_t_post
            predict_prior(h_t) → z_t_prior
            decode(h_t, z_t_post) → feat_t_recon
            reward_head(h_t, z_t_post) → reward_pred

  Curiosity: ensemble_prior(h_t) → disagreement → intrinsic reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_gaussian(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return mean + torch.randn_like(std) * std


class RSSM(nn.Module):
    def __init__(self, obs_dim: int = 33, action_dim: int = 3,
                 feat_dim: int = 64, deter_dim: int = 64, stoch_dim: int = 32,
                 n_ensemble: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # Encoder: percepta features → deterministic latent h
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, deter_dim),
        )

        # Posterior: (h, percepta_features) → stochastic z
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + feat_dim, 64), nn.ReLU(),
        )
        self.post_mean = nn.Linear(64, stoch_dim)
        self.post_std = nn.Linear(64, stoch_dim)

        # Prior: h → stochastic z (predicted without observation)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 64), nn.ReLU(),
        )
        self.prior_mean = nn.Linear(64, stoch_dim)
        self.prior_std = nn.Linear(64, stoch_dim)

        # Ensemble prior heads (for curiosity via disagreement)
        self.ensemble_prior_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(deter_dim, 32), nn.ReLU(),
                nn.Linear(32, stoch_dim * 2),
            ) for _ in range(n_ensemble)
        ])

        # Recurrent model
        self.gru_cell = nn.GRUCell(deter_dim + stoch_dim + action_dim, deter_dim)

        # JEPA predictor (was decoder): (h, z) → predicted features
        self.jepa_predictor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 128), nn.ReLU(),
            nn.Linear(128, feat_dim),
        )

        # Reward head: (h, z) → scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Goal proximity head: (h, z) → -distance_to_goal (for MPC planning)
        # Higher = closer to goal. Trained via regression on episode data.
        self.goal_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_obs(self, feat: torch.Tensor) -> torch.Tensor:
        """Encode features into deterministic latent h."""
        return self.encoder(feat)

    def infer_posterior(self, h: torch.Tensor, feat: torch.Tensor):
        x = torch.cat([h, feat], dim=-1)
        x = self.post_net(x)
        mean = self.post_mean(x)
        std = F.softplus(self.post_std(x)) + 0.01
        return _sample_gaussian(mean, std), mean, std

    def predict_prior(self, h: torch.Tensor):
        x = self.prior_net(h)
        mean = self.prior_mean(x)
        std = F.softplus(self.prior_std(x)) + 0.01
        return _sample_gaussian(mean, std), mean, std

    def recurrent_step(self, h: torch.Tensor, z: torch.Tensor,
                       action: torch.Tensor) -> torch.Tensor:
        return self.gru_cell(torch.cat([h, z, action], dim=-1), h)

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.jepa_predictor(torch.cat([h, z], dim=-1))

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([h, z], dim=-1)).squeeze(-1)

    def predict_goal_proximity(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict -distance_to_goal from latent state. Higher = closer to goal."""
        return self.goal_head(torch.cat([h, z], dim=-1)).squeeze(-1)

    def ensemble_curiosity(self, h: torch.Tensor):
        """Ensemble disagreement as curiosity signal.

        Returns:
          curiosity: (B,) average variance across ensemble predictions
          z_ensemble: (E, B, D) ensemble samples
        """
        z_list = []
        for net in self.ensemble_prior_nets:
            out = net(h)
            mean, log_std = out.chunk(2, dim=-1)
            std = F.softplus(log_std) + 0.01
            z = _sample_gaussian(mean, std)
            z_list.append(z.unsqueeze(0))

        z_ensemble = torch.cat(z_list, dim=0)
        var = z_ensemble.var(dim=0, unbiased=False).mean(dim=-1)
        return var, z_ensemble

    def forward_sequence(self, feats: torch.Tensor, actions: torch.Tensor,
                         rewards: torch.Tensor,
                         goal_dists: torch.Tensor | None = None) -> dict:
        """Process a full sequence with JEPA objective.

        JEPA: Predicts PerceptaModel features in latent space (no raw obs reconstruction).
        Targets are stop_grad(feats) to decouple from PerceptaModel training.

        feats:   (T, B, feat_dim) PerceptaModel features
        actions: (T, B, action_dim)
        rewards: (T, B)

        Returns dict with latents, predictions, and loss components.
        """
        T, B = feats.shape[:2]
        device = feats.device
        recon_target = feats.detach()  # JEPA: stop gradient on targets

        # Pre-allocate storage
        h = torch.zeros(B, self.deter_dim, device=device)
        z_post = torch.zeros(B, self.stoch_dim, device=device)
        z_prior = torch.zeros(B, self.stoch_dim, device=device)

        h_all = []
        z_post_all, z_prior_all = [], []
        recon_all, reward_pred_all, curiosity_all = [], [], []
        goal_prox_all = []

        kl_sum = 0.0
        recon_sum = 0.0
        reward_sum = 0.0

        for t in range(T):
            if t == 0:
                # Initial step: encode directly (no previous action)
                h = self.encode_obs(feats[t])
            else:
                # Recurrent: from previous (h, z, action)
                h = self.recurrent_step(h, z_post, actions[t - 1])

            # Posterior (informed by observation)
            z_post, pm, ps = self.infer_posterior(h, feats[t])

            # Prior (predicted without observation)
            z_prior, prm, prs = self.predict_prior(h)

            # Reconstruction and reward
            recon = self.decode(h, z_post)
            rew_pred = self.predict_reward(h, z_post)

            # Curiosity
            curiosity, _ = self.ensemble_curiosity(h)

            # Goal proximity prediction (for MPC planning)
            goal_prox = self.predict_goal_proximity(h, z_post)

            # Losses (accumulated)
            kl_sum += self._kl_div(pm, ps, prm, prs).sum()
            recon_sum += F.mse_loss(recon, recon_target[t],
                                    reduction='none').sum(dim=-1).sum()
            reward_sum += F.mse_loss(rew_pred, rewards[t], reduction='none').sum()

            # Store
            h_all.append(h)
            z_post_all.append(z_post)
            z_prior_all.append(z_prior)
            recon_all.append(recon)
            reward_pred_all.append(rew_pred)
            curiosity_all.append(curiosity)
            goal_prox_all.append(goal_prox)

        n = T * B
        result = {
            'h': torch.stack(h_all),
            'z_post': torch.stack(z_post_all),
            'z_prior': torch.stack(z_prior_all),
            'obs_recon': torch.stack(recon_all),
            'reward_pred': torch.stack(reward_pred_all),
            'curiosity': torch.stack(curiosity_all),
            'goal_prox': torch.stack(goal_prox_all),
            'kl_loss': kl_sum / n,
            'recon_loss': recon_sum / n,
            'reward_loss': reward_sum / n,
        }

        # Goal head loss (optional, for MPC planning training)
        if goal_dists is not None:
            goal_prox_all = torch.stack(goal_prox_all)
            goal_target = -goal_dists  # we predict -distance (higher = closer)
            result['goal_loss'] = F.mse_loss(goal_prox_all, goal_target, reduction='none').sum() / n
        else:
            result['goal_loss'] = torch.tensor(0.0, device=device)

        return result

    def imagine_sequence(self, h_start: torch.Tensor, z_start: torch.Tensor,
                         actor_policy: callable, horizon: int,
                         deterministic: bool = False) -> dict:
        """Generate imagined trajectory from initial state.

        Args:
          h_start: (B, deter_dim)
          z_start: (B, stoch_dim)
          actor_policy: callable(h, z) → action (B, action_dim)
          horizon: number of steps to imagine
          deterministic: if True, use mean instead of sample for prior

        Returns dict of imagined h, z, action, reward sequences.
        """
        B = h_start.shape[0]
        h, z = h_start, z_start

        h_all, z_all, action_all, reward_all = [], [], [], []

        for _ in range(horizon):
            action = actor_policy(h, z)

            h = self.recurrent_step(h, z, action)
            if deterministic:
                _, z_mean, _ = self.predict_prior(h)
                z = z_mean
            else:
                z, _, _ = self.predict_prior(h)

            reward = self.predict_reward(h, z)

            h_all.append(h)
            z_all.append(z)
            action_all.append(action)
            reward_all.append(reward)

        return {
            'h': torch.stack(h_all),
            'z': torch.stack(z_all),
            'action': torch.stack(action_all),
            'reward': torch.stack(reward_all),
        }

    @staticmethod
    def _kl_div(mu1, sigma1, mu2, sigma2):
        return torch.log(sigma2 / sigma1 + 1e-8) + \
               (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2 + 1e-8) - 0.5
