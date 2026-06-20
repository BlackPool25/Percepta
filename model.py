import torch
import torch.nn as nn
import torch.nn.functional as F


class FastDecayPath(nn.Module):
    """One fast-weight path with intelligent, usage-weighted decay.

    Each hidden unit (neuron) has a continuous stability score tracking
    its mean activation magnitude over time. Higher stability = slower
    decay toward initial values. This is metaplasticity: the unit's
    "plasticity" (how easily it changes) is inversely proportional to
    its recent contribution.

    Decay pushes weights toward their initialization, not toward zero,
    so the unit retains the capacity to reactivate when needed.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 decay_rate: float, path_name: str):
        super().__init__()
        self.decay_rate = decay_rate
        self.path_name = path_name
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.register_buffer('stability', torch.zeros(hidden_dim))
        self._register_init('fc_in.weight', self.fc_in.weight.data.clone())
        self._register_init('fc_in.bias', self.fc_in.bias.data.clone())
        self._register_init('fc_out.weight', self.fc_out.weight.data.clone())
        self._register_init('fc_out.bias', self.fc_out.bias.data.clone())

    def _register_init(self, name: str, tensor: torch.Tensor):
        self.register_buffer(f'init_{name.replace(".", "_")}', tensor.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_in(x))
        self._last_h = h.detach()
        return self.fc_out(h)

    @torch.no_grad()
    def update_stability(self):
        """Accumulate continuous activation magnitude per hidden unit."""
        mag = self._last_h.abs().mean(dim=0)
        self.stability += mag

    @torch.no_grad()
    def decay_step(self):
        """Decay toward initial values. Per-unit rate inversely prop to stability.

        Fully vectorized — no loops.
        """
        factor = self.decay_rate / (1.0 + self.stability * 0.01)

        diff_in = self.init_fc_in_weight - self.fc_in.weight.data
        self.fc_in.weight.data.add_(diff_in * factor.view(-1, 1))

        diff_bias = self.init_fc_in_bias - self.fc_in.bias.data
        self.fc_in.bias.data.add_(diff_bias * factor)

        diff_out = self.init_fc_out_weight - self.fc_out.weight.data
        self.fc_out.weight.data.add_(diff_out * factor.view(1, -1))

        diff_out_bias = self.init_fc_out_bias - self.fc_out.bias.data
        self.fc_out.bias.data.add_(diff_out_bias * factor.mean())


class BiDirSlowCNN(nn.Module):
    """Bi-directional fast/slow CNN with three fast pathways.

    Slow path: conv1→conv2→pool→fc1→fc2 (consolidated, protected).
    Fast paths (3): from shared base features, with different decay rates.
      - General (128 units, slow decay): broad recurring patterns
      - Specific (32 units, medium decay): episode-level details
      - Residual (16 units, fast decay): edge cases, error correction

    Bi-directional control: slow layer's softmax confidence modulates
    the effective learning rate of each fast path. Higher slow confidence
    = slower fast adaptation (don't override what's already known).
    """
    def __init__(self, k_wta: int = 0, fast_gen_dim: int = 128,
                 fast_spec_dim: int = 32, fast_resid_dim: int = 16,
                 fast_lr_base: float = 0.01):
        super().__init__()
        self.k_wta = k_wta
        self.fast_lr_base = fast_lr_base

        base_dim = 14 * 14 * 64
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(base_dim, 256)
        self.fc2 = nn.Linear(256, 10)

        self.fast_gen = FastDecayPath(base_dim, fast_gen_dim, 10, 0.005, 'gen')
        self.fast_spec = FastDecayPath(base_dim, fast_spec_dim, 10, 0.05, 'spec')
        self.fast_resid = FastDecayPath(base_dim, fast_resid_dim, 10, 0.2, 'resid')

    def _base_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._base_features(x)

        h = F.relu(self.fc1(base))
        if self.k_wta > 0 and self.training:
            _, topk_idx = torch.topk(h, self.k_wta, dim=1)
            mask = torch.zeros_like(h).scatter_(1, topk_idx, 1.0)
            h = h * mask
        slow_out = self.fc2(h)

        # Bi-directional confidence signal (inverse entropy, scaled [0,1])
        with torch.no_grad():
            probs = F.softmax(slow_out, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            self._batch_confidence = torch.exp(-entropy).mean().item()

        gen_out = self.fast_gen(base)
        spec_out = self.fast_spec(base)
        resid_out = self.fast_resid(base)

        return slow_out + gen_out + spec_out + resid_out

    def get_confidence(self) -> float:
        """Slow-layer batch confidence, used by training loop for fast LR modulation."""
        return getattr(self, '_batch_confidence', 0.5)

    @torch.no_grad()
    def update_fast_stabilities(self):
        for path in [self.fast_gen, self.fast_spec, self.fast_resid]:
            path.update_stability()

    @torch.no_grad()
    def decay_fast_weights(self):
        for path in [self.fast_gen, self.fast_spec, self.fast_resid]:
            path.decay_step()

    def get_slow_params(self):
        return (list(self.conv1.parameters()) + list(self.conv2.parameters()) +
                list(self.fc1.parameters()) + list(self.fc2.parameters()))

    def get_fast_params(self):
        return (list(self.fast_gen.parameters()) + list(self.fast_spec.parameters()) +
                list(self.fast_resid.parameters()))


class SlowCNN(nn.Module):
    """Legacy model — no fast layer. Used for naive/EWC baselines."""
    def __init__(self, k_wta: int = 0):
        super().__init__()
        self.k_wta = k_wta
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(14 * 14 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.k_wta > 0 and self.training:
            _, topk_idx = torch.topk(x, self.k_wta, dim=1)
            mask = torch.zeros_like(x).scatter_(1, topk_idx, 1.0)
            x = x * mask
        x = self.fc2(x)
        return x

    def get_slow_params(self):
        return list(self.parameters())

    def get_confidence(self) -> float:
        return 0.5


def create_model(k_wta: int = 0, use_fast_layer: bool = False,
                 fast_gen_dim: int = 128, fast_spec_dim: int = 32,
                 fast_resid_dim: int = 16, fast_lr_base: float = 0.01) -> nn.Module:
    if use_fast_layer:
        return BiDirSlowCNN(k_wta=k_wta, fast_gen_dim=fast_gen_dim,
                            fast_spec_dim=fast_spec_dim,
                            fast_resid_dim=fast_resid_dim,
                            fast_lr_base=fast_lr_base)
    return SlowCNN(k_wta=k_wta)
