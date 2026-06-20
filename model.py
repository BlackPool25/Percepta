import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FastDecayPath(nn.Module):
    """Fast-weight path with usage-weighted decay. One of three in the fast system."""
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


class SlowResNet(nn.Module):
    """Shared backbone: conv1→bn1→relu→layer1→layer2 (adapted for CIFAR)."""
    def __init__(self):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        # CIFAR adaptation: 3×3 conv stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1  # 2 blocks, 64→64
        self.layer2 = base.layer2  # 2 blocks, 64→128
        self._out_dim = 128 * 16 * 16  # 128ch × 16×16 (CIFAR: 32→layer1→32→layer2→16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # 32→16
        x = self.layer2(x)  # 16→8
        return x  # (B, 128, 8, 8)

    def forward_flatten(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).view(x.size(0), -1)


class BiDirSlowResNet(nn.Module):
    """Bi-directional fast/slow ResNet with three fast pathways.

    Shared: conv1→bn1→relu→layer1→layer2 (feature extraction).
    Slow path: layer3→layer4→avgpool→fc (consolidated knowledge).
    Fast paths (3): parallel adapters from shared features.

    Slow confidence gates fast learning rate (bi-directional).
    Fast decay is usage-weighted (metaplasticity).
    """
    def __init__(self, fast_gen_dim: int = 128, fast_spec_dim: int = 32,
                 fast_resid_dim: int = 16, fast_lr_base: float = 0.01,
                 num_classes: int = 10):
        super().__init__()
        self.fast_lr_base = fast_lr_base

        # Shared feature extraction
        self.shared = SlowResNet()

        # Slow path (consolidated)
        base = torchvision.models.resnet18(weights=None)
        self.slow_layer3 = base.layer3  # 2 blocks, 128→256
        self.slow_layer4 = base.layer4  # 2 blocks, 256→512
        self.slow_avgpool = base.avgpool
        self.slow_fc = nn.Linear(512, num_classes)

        # Fast paths from shared features (128 * 8 * 8 = 8192-dim)
        fdim = self.shared._out_dim
        self.fast_gen = FastDecayPath(fdim, fast_gen_dim, num_classes, 0.005, 'gen')
        self.fast_spec = FastDecayPath(fdim, fast_spec_dim, num_classes, 0.05, 'spec')
        self.fast_resid = FastDecayPath(fdim, fast_resid_dim, num_classes, 0.2, 'resid')

    def forward(self, x: torch.Tensor, buffer=None, retrieval_k: int = 0
                ) -> torch.Tensor:
        # Shared features
        shared_feat = self.shared.forward(x)  # (B, 128, 8, 8)
        shared_flat = shared_feat.view(x.size(0), -1)  # (B, 8192)

        # Slow path
        h = self.slow_layer3(shared_feat)
        h = self.slow_layer4(h)
        h = self.slow_avgpool(h)
        h = h.view(x.size(0), -1)
        slow_out = self.slow_fc(h)

        # Bi-directional confidence signal
        with torch.no_grad():
            probs = F.softmax(slow_out, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            self._batch_confidence = torch.exp(-entropy).mean().item()

        # Fast paths
        gen_out = self.fast_gen(shared_flat)
        spec_out = self.fast_spec(shared_flat)
        resid_out = self.fast_resid(shared_flat)

        # Episodic retrieval bias
        episodic_bias = 0
        if buffer is not None and retrieval_k > 0 and self._batch_confidence < 0.85:
            q_feat = shared_flat[0:1].squeeze(0)
            results = buffer.retrieve(q_feat, k=retrieval_k, use_core=True, device=x.device)
            if results:
                retrieved_logits = []
                for r_in, r_lbl, r_feat in results:
                    # Retrieve shared features for this memory
                    with torch.no_grad():
                        r_shared = self.shared.forward(r_in)
                        r_flat = r_shared.view(1, -1)
                        r_g = self.fast_gen(r_flat)
                        r_s = self.fast_spec(r_flat)
                        r_r = self.fast_resid(r_flat)
                        retrieved_logits.append(r_g + r_s + r_r)
                episodic_bias = torch.stack(retrieved_logits).mean(dim=0)

        return slow_out + gen_out + spec_out + resid_out + episodic_bias

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
        return (list(self.shared.parameters()) +
                list(self.slow_layer3.parameters()) +
                list(self.slow_layer4.parameters()) +
                list(self.slow_avgpool.parameters()) +
                list(self.slow_fc.parameters()))

    def get_fast_params(self):
        return (list(self.fast_gen.parameters()) +
                list(self.fast_spec.parameters()) +
                list(self.fast_resid.parameters()))

    def extract_fc1_features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared features used as 'features' for retrieval matching."""
        return self.shared.forward_flatten(x)


class SlowCNN(nn.Module):
    """Legacy 2-conv CNN for MNIST-scale tasks. Kept for backward compat."""
    def __init__(self, k_wta: int = 0, in_channels: int = 1, input_hw: int = 28):
        super().__init__()
        self.k_wta = k_wta
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        base_dim = 64 * (input_hw // 2) * (input_hw // 2)
        self.fc1 = nn.Linear(base_dim, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_slow_params(self):
        return list(self.parameters())

    def get_confidence(self) -> float:
        return 0.5


def create_model(k_wta: int = 0, use_fast_layer: bool = False,
                 fast_gen_dim: int = 128, fast_spec_dim: int = 32,
                 fast_resid_dim: int = 16, fast_lr_base: float = 0.01,
                 in_channels: int = 1, input_hw: int = 28,
                 use_resnet: bool = False,
                 num_classes: int = 10) -> nn.Module:
    if use_resnet:
        if use_fast_layer:
            return BiDirSlowResNet(
                fast_gen_dim=fast_gen_dim, fast_spec_dim=fast_spec_dim,
                fast_resid_dim=fast_resid_dim, fast_lr_base=fast_lr_base,
                num_classes=num_classes,
            )
        # ResNet without fast paths (for naive/EWC baselines on CIFAR)
        # Use the standard ResNet-18 adapted for CIFAR
        base = torchvision.models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, num_classes)
        return base
    if use_fast_layer:
        return SlowCNN(k_wta=k_wta, in_channels=in_channels,
                       input_hw=input_hw)
    return SlowCNN(k_wta=k_wta, in_channels=in_channels, input_hw=input_hw)
