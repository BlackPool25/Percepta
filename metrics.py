import torch
from torch.utils.data import DataLoader
import torch.nn as nn


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def compute_acc(
    model: nn.Module,
    tasks: list[tuple[DataLoader, DataLoader]],
    up_to_task: int,
    device: torch.device,
) -> float:
    accs = []
    for task_id in range(up_to_task + 1):
        _, test_loader = tasks[task_id]
        acc = evaluate(model, test_loader, device)
        accs.append(acc)
    return sum(accs) / len(accs)


def compute_bwt(
    task_accs_now: list[float],
    task_accs_after_learning: list[float],
) -> float:
    diffs = [
        task_accs_now[i] - task_accs_after_learning[i]
        for i in range(len(task_accs_now))
    ]
    return sum(diffs) / len(diffs)
