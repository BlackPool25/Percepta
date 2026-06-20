import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

from metrics import evaluate, compute_acc


def compute_fisher_diag(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.train()
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    criterion = nn.CrossEntropyLoss()

    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2
        n += x.size(0)

    for name in fisher:
        fisher[name] /= max(n, 1)
    return fisher


def get_layer_name(param_name: str) -> str:
    return param_name.rsplit('.', 1)[0] if '.' in param_name else param_name


def normalize_fisher_mask(mask: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-layer normalization: each layer's max Fisher is scaled to 1.0.
    
    Preserves within-layer ranking and cross-layer comparability
    (each layer contributes equally to the EWC penalty by default).
    """
    normalized = {}
    layer_maxes: dict[str, float] = {}
    for name, tensor in mask.items():
        layer = get_layer_name(name)
        m = tensor.max().item()
        layer_maxes[layer] = max(layer_maxes.get(layer, 0.0), m)
    for name, tensor in mask.items():
        layer = get_layer_name(name)
        denom = layer_maxes.get(layer, 1.0)
        if denom > 0:
            normalized[name] = tensor / denom
        else:
            normalized[name] = tensor.clone()
    return normalized


def merge_fisher_masks(
    fisher_masks: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not fisher_masks:
        return {}
    merged = {name: torch.zeros_like(mask) for name, mask in fisher_masks[0].items()}
    for mask in fisher_masks:
        for name in merged:
            merged[name] = torch.max(merged[name], mask[name])
    return merged


def ewc_penalty(
    model: nn.Module,
    fisher_masks: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    theta_star: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    lambda_ewc: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    if isinstance(fisher_masks, dict):
        fisher_list = [fisher_masks]
        theta_list = [theta_star]
    else:
        fisher_list = fisher_masks
        theta_list = theta_star
    if not fisher_list:
        return torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    for fm, ts in zip(fisher_list, theta_list):
        for name, param in model.named_parameters():
            if name in fm:
                total += (fm[name] * (param - ts[name]) ** 2).sum()
    return lambda_ewc * total


def train_naive(
    model: nn.Module,
    tasks: list[tuple[DataLoader, DataLoader]],
    epochs_per_task: int,
    lr: float,
    device: torch.device,
    momentum: float = 0.9,
) -> list[dict]:
    history = []
    task_accs_after: list[float] = []
    for task_id, (train_loader, test_loader) in enumerate(tasks):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs_per_task):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        # Record accuracy right after learning this task
        task_accs_after.append(evaluate(model, test_loader, device))
        per_task_now = [evaluate(model, tasks[t][1], device) for t in range(task_id + 1)]
        from metrics import compute_bwt
        bwt = compute_bwt(per_task_now, task_accs_after)
        history.append({
            'task_id': task_id,
            'acc': compute_acc(model, tasks, task_id, device),
            'bwt': bwt,
            'per_task_accs': per_task_now,
        })
    return history


def train_ewc(
    model: nn.Module,
    tasks: list[tuple[DataLoader, DataLoader]],
    epochs_per_task: int,
    lr: float,
    lambda_ewc: float,
    device: torch.device,
    momentum: float = 0.9,
) -> list[dict]:
    fisher_masks_list: list[dict[str, torch.Tensor]] = []
    theta_star_list: list[dict[str, torch.Tensor]] = []

    history = []
    task_accs_after: list[float] = []
    for task_id, (train_loader, test_loader) in enumerate(tasks):
        # Snapshot parameters BEFORE training this task
        theta_before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        for epoch in range(epochs_per_task):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                if fisher_masks_list:
                    loss += ewc_penalty(
                        model, fisher_masks_list, theta_star_list, lambda_ewc
                    )
                loss.backward()
                optimizer.step()

        # Compute Fisher on THIS task's data using the POST-TRAINING model
        # This tells us: which weights are important for task_id?
        # The Fisher is the diagonal of the expected squared gradient of log p(y|x)
        fisher_task = compute_fisher_diag(model, train_loader, device)
        fisher_task = normalize_fisher_mask(fisher_task)
        fisher_masks_list.append(fisher_task)
        theta_star_list.append({
            name: param.detach().clone()
            for name, param in model.named_parameters()
        })

        # Record metrics
        task_accs_after.append(evaluate(model, test_loader, device))
        per_task_now = [evaluate(model, tasks[t][1], device) for t in range(task_id + 1)]
        from metrics import compute_bwt
        bwt = compute_bwt(per_task_now, task_accs_after)
        history.append({
            'task_id': task_id,
            'acc': compute_acc(model, tasks, task_id, device),
            'bwt': bwt,
            'per_task_accs': per_task_now,
        })
    return history
