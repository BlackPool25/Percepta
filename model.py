import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowCNN(nn.Module):
    """Slow (cortex) network: standard CNN, protected by replay + gate.

    When used standalone (no fast layer), get_slow_params() returns all params.
    """
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

    def get_slow_params(self) -> list[torch.Tensor]:
        return list(self.parameters())


class FastWeightAdapter(nn.Module):
    """Small, fast-adapting parallel network (hippocampus analogue).

    Learns to predict the RESIDUAL of the slow network's output.
    When slow_out is already correct, gradient for fast_out is ~0,
    so the fast layer naturally disables itself — it can't harm.

    Get its input from base features (after conv2+pool), not the slow
    layer's hidden representation — fully independent feature pathway.
    """
    def __init__(self, input_dim: int = 14 * 14 * 64,
                 hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class SlowCNNWithFast(nn.Module):
    """Slow network with parallel fast-weight adapter.

    Slow path: conv1→conv2→pool→fc1→fc2 (protected, consolidated).
    Fast path: conv1→conv2→pool→fast_fc1→fast_fc2 (ephemeral, reset each epoch).

    Output = slow_out + fast_out  (residual ensemble).
    Fast layer sees the same base features but through a small independent network.
    """
    def __init__(self, k_wta: int = 0, fast_hidden: int = 64,
                 fast_lr: float = 0.01):
        super().__init__()
        self.k_wta = k_wta
        self.fast_lr = fast_lr

        # Shared feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Slow path
        self.fc1 = nn.Linear(14 * 14 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

        # Fast path (parallel, independent)
        self.fast = FastWeightAdapter(
            input_dim=14 * 14 * 64,
            hidden_dim=fast_hidden,
            output_dim=10,
        )

    def _base_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._base_features(x)

        # Slow path
        h = F.relu(self.fc1(base))
        if self.k_wta > 0 and self.training:
            _, topk_idx = torch.topk(h, self.k_wta, dim=1)
            mask = torch.zeros_like(h).scatter_(1, topk_idx, 1.0)
            h = h * mask
        slow_out = self.fc2(h)

        # Fast path (residual correction — not redundant)
        fast_out = self.fast(base)

        return slow_out + fast_out

    def reset_fast_weights(self):
        self.fast.reset_parameters()

    def get_slow_params(self) -> list[torch.Tensor]:
        return (list(self.conv1.parameters()) + list(self.conv2.parameters()) +
                list(self.fc1.parameters()) + list(self.fc2.parameters()))

    def get_fast_params(self) -> list[torch.Tensor]:
        return list(self.fast.parameters())


def create_model(k_wta: int = 0, use_fast_layer: bool = False,
                 fast_hidden: int = 64, fast_lr: float = 0.01) -> nn.Module:
    if use_fast_layer:
        return SlowCNNWithFast(k_wta=k_wta, fast_hidden=fast_hidden, fast_lr=fast_lr)
    return SlowCNN(k_wta=k_wta)
