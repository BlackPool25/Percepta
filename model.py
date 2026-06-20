"""Vector-observation fast/slow PerceptaModel for continuous control.

Architecture:
  obs (33-dim) → shared MLP → features
  task (3-dim) → task_encoder → task_features
  features + task_features → slow_path + 3 fast_paths (parallel)
  combined = slow_out + gen_out + spec_out + resid_out → RSSM
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
    """MLP block with residual connection and layer norm."""
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
    """Fast/slow perception encoder with deep residual MLP backbone.

    Architecture:
      obs → shared_in → [ResBlock×N] → shared_out → feat (64-dim)
      task → task_enc → task_feat (16-dim)
      feat + task → slow path (stable consolidated knowledge)
                 → fast paths (×3, adaptive with decay)

    Combined = slow_out + gen_out + spec_out + resid_out → RSSM
    """
    def __init__(self, obs_dim: int = 33, task_dim: int = 3, feat_dim: int = 64,
                 hidden_dim: int = 256, n_res_blocks: int = 3,
                 fast_gen_dim: int = 64, fast_spec_dim: int = 32,
                 fast_resid_dim: int = 16):
        super().__init__()
        self.feat_dim = feat_dim

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

        # Task context encoder (separate head)
        self.task_enc = nn.Sequential(
            nn.Linear(task_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

        # Combined dimension: obs_features + task_features
        combined_dim = feat_dim + 16

        # Slow path (consolidated knowledge, stable)
        self.slow = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

        # Fast paths (parallel, adaptive with different decay rates)
        self.fast_gen = FastDecayPath(combined_dim, fast_gen_dim, feat_dim, 0.005, 'gen')
        self.fast_spec = FastDecayPath(combined_dim, fast_spec_dim, feat_dim, 0.05, 'spec')
        self.fast_resid = FastDecayPath(combined_dim, fast_resid_dim, feat_dim, 0.2, 'resid')

        # Autoencoding decoder (for self-supervised training)
        # Takes combined features → reconstructs observation
        self.ae_decoder = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feat_dim),
        )
        self.ae_obs_head = nn.Linear(feat_dim, obs_dim)

    def forward(self, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """Encode obs → combined features (used downstream by RSSM)."""
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

        with torch.no_grad():
            slow_norm = slow_out.norm(dim=1).mean()
            self._batch_confidence = torch.sigmoid(slow_norm * 0.1).item()

        return slow_out + gen_out + spec_out + resid_out

    def forward_with_ae(self, obs: torch.Tensor, task: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with autoencoding reconstruction.

        Returns: (features, reconstructed_obs)
        """
        feat = self.forward(obs, task)
        recon_h = self.ae_decoder(feat)
        recon = self.ae_obs_head(recon_h)
        return feat, recon

    def ae_loss(self, obs: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        _, recon = self.forward_with_ae(obs, task)
        return F.mse_loss(recon, obs)

    def get_confidence(self) -> float:
        return getattr(self, '_batch_confidence', 0.5)

    @torch.no_grad()
    def update_fast_stabilities(self):
        for p in [self.fast_gen, self.fast_spec, self.fast_resid]:
            p.update_stability()

    @torch.no_grad()
    def decay_fast_weights(self):
        for p in [self.fast_gen, self.fast_spec, self.fast_resid]:
            p.decay_step()

    def get_slow_params(self):
        return (list(self.shared_in.parameters()) +
                list(self.shared_blocks.parameters()) +
                list(self.shared_out.parameters()) +
                list(self.task_enc.parameters()) +
                list(self.slow.parameters()))

    def get_fast_params(self):
        return (list(self.fast_gen.parameters()) +
                list(self.fast_spec.parameters()) +
                list(self.fast_resid.parameters()))
