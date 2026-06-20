import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowCNN(nn.Module):
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


def create_model(k_wta: int = 0) -> nn.Module:
    return SlowCNN(k_wta=k_wta)
