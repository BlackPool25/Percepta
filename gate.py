import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from baselines import compute_fisher_diag, merge_fisher_masks, ewc_penalty


def is_decreasing(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    xs = list(range(len(values)))
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(values)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, values))
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return False
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope < 0


def is_candidate(
    entry,
    freq_threshold: int = 50,
    persist_window: int = 10,
) -> bool:
    frequent = entry.seen_count >= freq_threshold
    if len(entry.pred_error_history) < 2:
        return frequent
    recent = entry.pred_error_history[-persist_window:]
    persistently_surprising = not is_decreasing(recent)
    return frequent or persistently_surprising


@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(model(inputs.to(device)), labels.to(device))
    return loss.item()


def _make_loader(
    inputs_list: list[torch.Tensor],
    labels_list: list[torch.Tensor],
    batch_size: int,
) -> DataLoader:
    all_inputs = torch.cat(inputs_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    dataset = TensorDataset(all_inputs, all_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def validate_and_commit(
    model: nn.Module,
    candidate,
    fisher_masks: dict[int, dict[str, torch.Tensor]],
    theta_star: dict[int, dict[str, torch.Tensor]],
    eps_gain: float,
    eps_forget: float,
    replay_loader: DataLoader,
    lr: float,
    steps: int,
    lambda_ewc: float,
    device: torch.device,
) -> bool:
    candidate_loader = _make_loader(
        candidate.inputs, candidate.labels, batch_size=128
    )

    # Measure old error on candidate data
    old_error = _eval_loss(
        model,
        torch.cat(candidate.inputs, dim=0),
        torch.cat(candidate.labels, dim=0),
        device,
    )

    # Measure old performance on replay sample (previously committed knowledge)
    old_replay_loss = 0.0
    if replay_loader is not None:
        for x, y in replay_loader:
            old_replay_loss += _eval_loss(model, x, y, device)
        old_replay_loss /= max(1, len(replay_loader))

    # Create shadow copy for tentative integration
    shadow = deepcopy(model).to(device)
    shadow_optimizer = torch.optim.SGD(shadow.parameters(), lr=lr, momentum=0.9)

    # Build merged Fisher + theta_star for EWC penalty during fine-tune
    if fisher_masks:
        committed_masks = merge_fisher_masks(list(fisher_masks.values()))
        committed_theta = theta_star_merged(list(theta_star.values()))
    else:
        committed_masks = {}
        committed_theta = {}

    # Build joint loader: candidate data + replay data interleaved
    candidate_x = torch.cat(candidate.inputs, dim=0)
    candidate_y = torch.cat(candidate.labels, dim=0)
    if replay_loader is not None:
        replay_blocks_x, replay_blocks_y = [], []
        for rx, ry in replay_loader:
            replay_blocks_x.append(rx)
            replay_blocks_y.append(ry)
        if replay_blocks_x:
            replay_x = torch.cat(replay_blocks_x, dim=0)
            replay_y = torch.cat(replay_blocks_y, dim=0)
            joint_x = torch.cat([candidate_x, replay_x], dim=0)
            joint_y = torch.cat([candidate_y, replay_y], dim=0)
        else:
            joint_x, joint_y = candidate_x, candidate_y
    else:
        joint_x, joint_y = candidate_x, candidate_y
    joint_loader = DataLoader(
        TensorDataset(joint_x, joint_y), batch_size=128, shuffle=True
    )

    # Fine-tune shadow on joint candidate + replay data
    criterion = nn.CrossEntropyLoss()
    for _ in range(steps):
        for x, y in joint_loader:
            x, y = x.to(device), y.to(device)
            shadow_optimizer.zero_grad()
            output = shadow(x)
            loss = criterion(output, y)
            if committed_masks:
                loss += ewc_penalty(shadow, committed_masks, committed_theta, lambda_ewc)
            loss.backward()
            shadow_optimizer.step()

    # Measure new error on candidate data
    new_error = _eval_loss(
        shadow,
        torch.cat(candidate.inputs, dim=0),
        torch.cat(candidate.labels, dim=0),
        device,
    )

    # Measure new forgetting on replay sample
    new_replay_loss = 0.0
    if replay_loader is not None:
        for x, y in replay_loader:
            new_replay_loss += _eval_loss(shadow, x, y, device)
        new_replay_loss /= max(1, len(replay_loader))

    gain = old_error - new_error
    forgetting = new_replay_loss - old_replay_loss

    if gain > eps_gain and forgetting < eps_forget:
        # Commit: copy shadow weights into model
        model.load_state_dict(shadow.state_dict())

        # Compute Fisher diagonal as byproduct (reuse gradients)
        fisher_current = compute_fisher_diag(model, candidate_loader, device)

        # Store
        fisher_masks[candidate.cluster_id] = fisher_current
        theta_star[candidate.cluster_id] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        return True

    return False


def theta_star_merged(
    theta_star_list: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not theta_star_list:
        return {}
    merged = {name: torch.zeros_like(t) for name, t in theta_star_list[0].items()}
    for ts in theta_star_list:
        for name in merged:
            merged[name] = merged[name] + ts[name]
    for name in merged:
        merged[name] /= len(theta_star_list)
    return merged
