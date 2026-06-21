"""PerceptaModel: fast/slow weight architecture that directly produces actions.

PerceptaModel IS the policy. No separate actor.
  obs + task → shared → slow (stable) + 3 fast paths (adaptive)
  → combined → action_head → action

Learning:
  - Fast paths: trained via gradient within episodes, decay toward init
  - Slow path: updated via gate consolidation (fast → slow transfer)
  - Autoencoding: self-supervised decoder keeps features informative
  - Confidence: slow path confidence gates fast path contribution
  - Metaplasticity: usage-weighted decay for fast paths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastDecayPath(nn.Module):
    """Fast-weight path with usage-weighted decay (metaplasticity).

    Maps to brain: fast-decaying synapses that consolidate with frequent use.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 decay_rate: float, path_name: str):
        super().__init__()
        self.decay_rate = decay_rate
        self.path_name = path_name
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._register_buffers()

    def _register_buffers(self):
        self.register_buffer('stability', torch.zeros(self.hidden_dim))
        for name, p in self.net.named_parameters():
            self.register_buffer(f'init_{name.replace(".", "_")}', p.data.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net[1](self.net[0](x))
        self._last_h = h.detach()
        return self.net[2](h)

    @torch.no_grad()
    def update_stability(self):
        mag = self._last_h.abs().mean(dim=0)
        self.stability += mag

    @torch.no_grad()
    def decay_step(self):
        factor = self.decay_rate / (1.0 + self.stability * 0.01)
        w = self.net[0].weight
        diff = self.init_0_weight - w
        w.data.add_(diff * factor.view(-1, 1))
        b = self.net[0].bias
        if b is not None:
            diff_b = self.init_0_bias - b
            b.data.add_(diff_b * factor)
        wo = self.net[2].weight
        diff_o = self.init_2_weight - wo
        wo.data.add_(diff_o * factor.view(1, -1))
        bo = self.net[2].bias
        if bo is not None:
            diff_ob = self.init_2_bias - bo
            bo.data.add_(diff_ob * factor.mean())


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PerceptaModel(nn.Module):
    """Fast/slow perception + action model.
    
    Architecture:
      obs + task → shared backbone → slow + 3 fast → combined → action
    
    Bi-directional confidence:
      - slow confidence γ = sigmoid(||slow_out|| * 0.1)
      - fast contribution scaled by (1 - γ)
      - When slow is confident, fast contributes little
      - When slow is uncertain, fast contributes strongly
    """
    def __init__(self, obs_dim=33, task_dim=3, action_dim=3,
                 feat_dim=64, hidden_dim=256, n_res_blocks=3,
                 fast_gen_dim=64, fast_spec_dim=32, fast_resid_dim=16,
                 spatial_dim=8):
        """spatial_dim: dimensions of explicit spatial features (computed, not learned)."""
        super().__init__()
        self.feat_dim = feat_dim
        self.spatial_dim = spatial_dim

        # Shared backbone: obs → residual blocks → features
        self.shared_in = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.shared_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim) for _ in range(n_res_blocks)
        ])
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

        # Task encoder (separate head)
        self.task_enc = nn.Sequential(
            nn.Linear(task_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

        combined_dim = feat_dim + 16

        # Slow path (consolidated knowledge)
        self.slow = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

        # Fast paths (×3, different decay rates)
        self.fast_gen = FastDecayPath(combined_dim, fast_gen_dim, feat_dim, 0.005, 'gen')
        self.fast_spec = FastDecayPath(combined_dim, fast_spec_dim, feat_dim, 0.05, 'spec')
        self.fast_resid = FastDecayPath(combined_dim, fast_resid_dim, feat_dim, 0.2, 'resid')

        # Action head: takes learned features + explicit spatial features
        action_input_dim = feat_dim + spatial_dim
        self.action_head = nn.Sequential(
            nn.Linear(action_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -1.5))

        # Autoencoding decoder (self-supervised: keeps features informative)
        self.ae_decoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
        )
        self.ae_obs_head = nn.Linear(feat_dim, obs_dim)

    def forward(self, obs: torch.Tensor, task: torch.Tensor):
        """Produce action from obs + task. Returns (action_mean, features).

        The action uses BOTH learned features (from PerceptaModel) and
        explicitly computed spatial features (relative positions).
        """
        # Learned features (from fast/slow paths)
        x = self.shared_in(obs)
        for block in self.shared_blocks:
            x = block(x)
        obs_feat = self.shared_out(x)
        task_feat = self.task_enc(task)
        combined = torch.cat([obs_feat, task_feat], dim=-1)

        slow_out = self.slow(combined)
        gen_out = self.fast_gen(combined)
        spec_out = self.fast_spec(combined)
        resid_out = self.fast_resid(combined)

        # Bi-directional confidence
        slow_norm = slow_out.norm(dim=1)
        conf = torch.sigmoid(slow_norm * 0.1).unsqueeze(1)
        self._batch_confidence = conf.mean().item()

        # Fast contribution scaled by (1 - confidence)
        fast_out = gen_out + spec_out + resid_out
        combined_out = slow_out + (1.0 - conf) * fast_out

        # Explicit spatial features
        spatial = self._compute_spatial_features(obs, task)

        # Action from combined learned + spatial features
        action_input = torch.cat([combined_out, spatial], dim=-1)
        action_mean = torch.tanh(self.action_head(action_input))

        # Features (for autoencoding, curiosity)
        features = slow_out + fast_out

        return action_mean, features

    def sample(self, obs: torch.Tensor, task: torch.Tensor,
               deterministic=False):
        """Sample action. Returns action tensor."""
        action_mean, _ = self.forward(obs, task)
        if deterministic:
            return action_mean
        std = F.softplus(self.action_log_std) + 0.01
        return torch.tanh(action_mean + torch.randn_like(std) * std)

    @staticmethod
    def _compute_spatial_features(obs: torch.Tensor, task: torch.Tensor):
        """Compute explicit spatial features from raw observation.

        Observation layout:
          0:2  agent_pos
          3:5  agent_vel
          6:8  obj1_pos, 9:11 obj1_vel
          12:14 obj2_pos, 15:17 obj2_vel
          18:20 obj3_pos, 21:23 obj3_vel
          24:26 goal_pos
          27:29 contacts
          30:32 task (one-hot: which object is target)

        Returns: [vec_to_target_x, vec_to_target_y,
                  vec_to_goal_x, vec_to_goal_y,
                  dist_to_target, dist_to_goal,
                  contact_with_target, target_in_front]
        """
        # Task: which object is the target
        task_oh = task  # one-hot, 3-dim
        B = obs.shape[0]
        device = obs.device

        # Agent position
        agent_pos = obs[:, 0:2]  # x, y

        # Goal position
        goal_pos = obs[:, 24:26]

        # Determine target object position based on task encoding
        # task is one-hot: [is_obj1, is_obj2, is_obj3]
        obj_positions = torch.stack([
            obs[:, 6:8],    # obj1
            obs[:, 12:14],  # obj2
            obs[:, 18:20],  # obj3
        ], dim=1)  # (B, 3, 2)

        # Weighted sum: pick the target object's position
        target_pos = (obj_positions * task_oh.unsqueeze(-1)).sum(dim=1)  # (B, 2)

        # Relative vectors
        vec_to_target = target_pos - agent_pos  # (B, 2)
        vec_to_goal = goal_pos - target_pos     # (B, 2)

        # Distances
        dist_to_target = torch.norm(vec_to_target, dim=1, keepdim=True)
        dist_to_goal = torch.norm(vec_to_goal, dim=1, keepdim=True)

        # Contact with target
        contacts = obs[:, 27:30]  # (B, 3) one-hot contacts
        contact_with_target = (contacts * task_oh).sum(dim=1, keepdim=True)

        # Is target in front? (same direction as agent's facing)
        # Simple heuristic: agent moving toward target
        vel = obs[:, 3:5]
        moving_toward = (vel * vec_to_target).sum(dim=1, keepdim=True) > 0
        target_in_front = moving_toward.float()

        return torch.cat([
            vec_to_target,       # 2
            vec_to_goal,         # 2
            dist_to_target,      # 1
            dist_to_goal,        # 1
            contact_with_target, # 1
            target_in_front,     # 1
        ], dim=1)  # total: 8 dims

    def get_features(self, obs: torch.Tensor, task: torch.Tensor):
        """Get features (for autoencoding, curiosity)."""
        _, features = self.forward(obs, task)
        return features

    def ae_loss(self, obs: torch.Tensor, task: torch.Tensor):
        """Self-supervised autoencoding loss."""
        _, features = self.forward(obs, task)
        recon = self.ae_obs_head(self.ae_decoder(features))
        return F.mse_loss(recon, obs)

    def get_confidence(self):
        return getattr(self, '_batch_confidence', 0.5)

    def update_fast_stabilities(self):
        for p in [self.fast_gen, self.fast_spec, self.fast_resid]:
            p.update_stability()

    def decay_fast_weights(self):
        for p in [self.fast_gen, self.fast_spec, self.fast_resid]:
            p.decay_step()

    def get_slow_params(self):
        return (list(self.shared_in.parameters()) +
                list(self.shared_blocks.parameters()) +
                list(self.shared_out.parameters()) +
                list(self.task_enc.parameters()) +
                list(self.slow.parameters()) +
                list(self.action_head.parameters()) +
                list(self.ae_decoder.parameters()) +
                list(self.ae_obs_head.parameters()))

    def get_fast_params(self):
        return (list(self.fast_gen.parameters()) +
                list(self.fast_spec.parameters()) +
                list(self.fast_resid.parameters()))
