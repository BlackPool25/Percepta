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
    # Only optimize slow params during validation — fast weights are ephemeral
    shadow_optimizer = torch.optim.SGD(shadow.get_slow_params(), lr=lr, momentum=0.9)

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


@torch.no_grad()
def _eval_accuracy(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    outputs = model(inputs.to(device))
    preds = outputs.argmax(dim=1)
    return (preds == labels.to(device)).float().mean().item()


def check_destabilization(
    model: nn.Module,
    buffer,
    committed_clusters: set,
    device: torch.device,
    destabilize_threshold: float = 0.80,
    destabilize_cooldown: int = 5,
) -> list[int]:
    """Check committed clusters for degraded core-set accuracy.

    Magnitude-scaled trigger:
    - < 50%: immediate destabilization
    - 50-80%: requires 3 consecutive detections

    Returns list of cluster IDs needing destabilization.
    """
    candidates = []
    for cid in sorted(committed_clusters):
        entry = buffer.entries.get(cid)
        if entry is None or not entry.committed:
            continue
        if not entry.core_inputs:
            continue

        inputs = torch.cat(entry.core_inputs, dim=0)
        labels = torch.cat(entry.core_labels, dim=0)
        acc = _eval_accuracy(model, inputs, labels, device)
        entry.core_acc_history.append(acc)

        # Cooldown check
        if entry.destabilize_count > 0:
            steps_since = len(entry.pred_error_history) - entry.last_destabilized_step
            if steps_since < destabilize_cooldown:
                continue

        # Magnitude-scaled trigger
        if acc < 0.50:
            candidates.append(cid)
        elif acc < destabilize_threshold:
            if len(entry.core_acc_history) >= 3:
                if all(a < destabilize_threshold for a in entry.core_acc_history[-3:]):
                    candidates.append(cid)

    return candidates


def destabilize_and_restabilize(
    model: nn.Module,
    buffer,
    cluster_id: int,
    device: torch.device,
    steps: int = 20,
    lr: float = 1e-4,
    eps_forget: float = 0.50,
) -> bool:
    """Destabilize a committed cluster, fine-tune on current data, restabilize.

    When a cluster's core-set accuracy degrades (concept drift), this:
    1. Removes cluster from replay pool (stop reinforcing old behavior)
    2. Fine-tunes on current buffer data (recent experiences)
    3. Checks collateral damage on other clusters
    4. Commits and restabilizes with fresh core-set
    """
    entry = buffer.entries.get(cluster_id)
    if entry is None or not entry.committed:
        return False
    if not entry.inputs:
        return False

    # Current data (post-drift experiences)
    current_x = torch.cat(entry.inputs, dim=0)
    current_y = torch.cat(entry.labels, dim=0)

    if current_x.size(0) < 5:
        return False

    # Measure old performance on OTHER committed clusters
    old_other_loss = 0.0
    other_count = 0
    for oid in buffer.entries:
        if oid == cluster_id or not buffer.entries[oid].committed:
            continue
        o_entry = buffer.entries[oid]
        if not o_entry.core_inputs:
            continue
        ox = torch.cat(o_entry.core_inputs, dim=0)
        oy = torch.cat(o_entry.core_labels, dim=0)
        old_other_loss += _eval_loss(model, ox, oy, device)
        other_count += 1
    old_other_loss /= max(other_count, 1)

    # Shadow fine-tune on current data only (no replay for this cluster)
    shadow = deepcopy(model).to(device)
    shadow_optimizer = torch.optim.SGD(shadow.get_slow_params(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    current_loader = DataLoader(
        TensorDataset(current_x, current_y), batch_size=128, shuffle=True
    )

    for _ in range(steps):
        for x, y in current_loader:
            x, y = x.to(device), y.to(device)
            shadow_optimizer.zero_grad()
            loss = criterion(shadow(x), y)
            loss.backward()
            shadow_optimizer.step()

    # Measure new performance on OTHER committed clusters
    new_other_loss = 0.0
    for oid in buffer.entries:
        if oid == cluster_id or not buffer.entries[oid].committed:
            continue
        o_entry = buffer.entries[oid]
        if not o_entry.core_inputs:
            continue
        ox = torch.cat(o_entry.core_inputs, dim=0)
        oy = torch.cat(o_entry.core_labels, dim=0)
        new_other_loss += _eval_loss(shadow, ox, oy, device)
        other_count += 1
    new_other_loss /= max(other_count, 1)

    forgetting = new_other_loss - old_other_loss
    if forgetting > eps_forget:
        return False

    # Commit
    model.load_state_dict(shadow.state_dict())
    if hasattr(model, 'reset_fast_weights'):
        model.reset_fast_weights()

    # Restabilize: update core-set from current data
    n = min(buffer.core_size_per_cluster, current_x.size(0))
    indices = torch.randperm(current_x.size(0))[:n]
    entry.core_inputs = [current_x[indices].cpu()]
    entry.core_labels = [current_y[indices].cpu()]
    entry.destabilize_count += 1
    entry.last_destabilized_step = len(entry.pred_error_history)
    entry.core_acc_history = []

    return True


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
