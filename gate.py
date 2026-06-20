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
    # Subsample candidate data to at most 256 images for efficiency
    all_cand_x = torch.cat(candidate.inputs, dim=0)
    all_cand_y = torch.cat(candidate.labels, dim=0)
    n_total = all_cand_x.size(0)
    if n_total > 256:
        perm = torch.randperm(n_total)[:256]
        all_cand_x = all_cand_x[perm]
        all_cand_y = all_cand_y[perm]
    candidate_ds = TensorDataset(all_cand_x, all_cand_y)
    candidate_loader = DataLoader(candidate_ds, batch_size=128, shuffle=True)

    # Measure old error on candidate data
    old_error = _eval_loss(model, all_cand_x, all_cand_y, device)

    # Measure old performance on replay sample (previously committed knowledge)
    old_replay_loss = 0.0
    if replay_loader is not None:
        for batch in replay_loader:
            x, y = batch[0], batch[1]
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

    if replay_loader is not None:
        replay_blocks_x, replay_blocks_y = [], []
        for batch in replay_loader:
            rx, ry = batch[0], batch[1]
            replay_blocks_x.append(rx)
            replay_blocks_y.append(ry)
        if replay_blocks_x:
            r_x = torch.cat(replay_blocks_x, dim=0)
            r_y = torch.cat(replay_blocks_y, dim=0)
            joint_x = torch.cat([all_cand_x, r_x], dim=0)
            joint_y = torch.cat([all_cand_y, r_y], dim=0)
        else:
            joint_x, joint_y = all_cand_x, all_cand_y
    else:
        joint_x, joint_y = all_cand_x, all_cand_y
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

    # Measure new error on candidate data (use same subset)
    new_error = _eval_loss(shadow, all_cand_x, all_cand_y, device)

    # Measure new forgetting on replay sample
    new_replay_loss = 0.0
    if replay_loader is not None:
        for batch in replay_loader:
            x, y = batch[0], batch[1]
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


# ─── Feature extraction helpers ────────────────────────────────────────────────

@torch.no_grad()
def _extract_features(model: nn.Module, inputs: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
    """Extract hidden activations (fc1 layer) for novelty detection."""
    model.eval()
    x = inputs.to(device)
    x = F.relu(model.conv1(x))
    x = F.relu(model.conv2(x))
    x = model.pool(x)
    x = x.view(x.size(0), -1)
    x = F.relu(model.fc1(x))
    return x.cpu()


@torch.no_grad()
def _compute_core_accuracy(model: nn.Module, entry, device: torch.device) -> float:
    """Accuracy on core-set."""
    if not entry.core_inputs:
        return 0.0
    model.eval()
    x = torch.cat(entry.core_inputs, dim=0).to(device)
    y = torch.cat(entry.core_labels, dim=0).to(device)
    preds = model(x).argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def _compute_novelty(model: nn.Module, entry, device: torch.device) -> float:
    """Feature-space distance between core-set and recent buffer data.

    Uses fc1 activations as the feature space.
    Compares centroid of core-set vs centroid of most recent buffer blocks.
    Returns cosine distance in [0, 2] where 0 = identical, >0.3 = novel.
    """
    if not entry.core_inputs or not entry.inputs:
        return 0.0

    core_x = torch.cat(entry.core_inputs, dim=0)
    core_feats = _extract_features(model, core_x, device)

    # Sample recent buffer data (last 25% of blocks, up to 200 examples)
    buf_blocks = entry.inputs
    n_recent = max(1, len(buf_blocks) // 4)
    recent_blocks = buf_blocks[-n_recent:]
    recent_x = torch.cat(recent_blocks, dim=0)
    if recent_x.size(0) > 200:
        perm = torch.randperm(recent_x.size(0))[:200]
        recent_x = recent_x[perm]
    buf_feats = _extract_features(model, recent_x, device)

    core_centroid = core_feats.mean(dim=0, keepdim=True)
    buf_centroid = buf_feats.mean(dim=0, keepdim=True)

    core_norm = core_centroid / (core_centroid.norm(p=2, dim=1, keepdim=True) + 1e-8)
    buf_norm = buf_centroid / (buf_centroid.norm(p=2, dim=1, keepdim=True) + 1e-8)

    cosine_sim = (core_norm @ buf_norm.t()).item()
    return 1.0 - cosine_sim


# ─── Destabilization v3 — 3-criteria detection + partial degradation ─────────

def check_destabilization_v3(
    model: nn.Module,
    buffer,
    committed_clusters: set,
    device: torch.device,
    commit_age_min: int = 3,
    acc_drop_margin: float = 0.15,
    novelty_threshold: float = 0.30,
    persist_checks: int = 3,
) -> dict[int, dict]:
    """Three-criteria destabilization detection.

    All three must fire:
    1. Prediction error (relative to commit-time accuracy)
    2. Novelty (feature-space distance from core-set to recent data)
    3. Persistence (pattern holds for N consecutive checks)

    Returns dict of {cluster_id: {metrics}} for candidates.
    """
    candidates = {}
    for cid in sorted(committed_clusters):
        entry = buffer.entries.get(cid)
        if entry is None or not entry.committed:
            continue
        if not entry.core_inputs:
            continue

        # Frequency gating: must have been committed long enough to stabilize
        steps_committed = len(entry.pred_error_history) - entry.last_destabilized_step
        if steps_committed < commit_age_min:
            continue

        # Cooldown gating: prevent rumination
        if entry.destabilize_count > 0:
            steps_since = len(entry.pred_error_history) - entry.last_destabilized_step
            if steps_since < commit_age_min * 2:
                continue

        # 1. Prediction error (relative to commit-time accuracy)
        commit_acc = entry.commit_accuracy
        current_acc = _compute_core_accuracy(model, entry, device)
        acc_drop = commit_acc - current_acc

        # 2. Novelty
        novelty = _compute_novelty(model, entry, device)

        # 3. Persistence
        entry.core_acc_history.append(current_acc)
        recent = entry.core_acc_history[-(persist_checks + 1):-1]
        is_persistent = len(recent) >= persist_checks and all(
            entry.commit_accuracy - a > acc_drop_margin * 0.5 for a in recent
        )

        if acc_drop > acc_drop_margin and novelty > novelty_threshold and is_persistent:
            # Compute contradiction strength for partial degradation
            candidates[cid] = {
                'acc_drop': acc_drop,
                'novelty': novelty,
                'current_acc': current_acc,
                'commit_acc': commit_acc,
            }

    return candidates


def destabilize_partial(
    model: nn.Module,
    buffer,
    cluster_id: int,
    importance_masks: dict,
    theta_star: dict,
    device: torch.device,
    steps: int = 20,
    lr: float = 1e-4,
    eps_forget: float = 0.50,
    candidate_info: dict | None = None,
) -> bool:
    """Destabilize with partial trace degradation (UPS analogue).

    Phase 1 — DEGRADE: Scale down old importance mask and theta_star
    by the contradiction strength. Strong contradiction (50%+ drop) →
    near-complete degradation. Weak contradiction → partial retention.

    Phase 2 — FINE-TUNE: Train on current buffer data without old protection.

    Phase 3 — RESTABILIZE: Compute fresh importance and core-set.
    """
    entry = buffer.entries.get(cluster_id)
    if entry is None or not entry.committed:
        return False
    if not entry.inputs:
        return False

    current_x = torch.cat(entry.inputs, dim=0)
    current_y = torch.cat(entry.labels, dim=0)
    if current_x.size(0) < 5:
        return False

    # Compute degradation strength from contradiction magnitude
    acc_drop = candidate_info['acc_drop'] if candidate_info else 0.3
    strength = min(1.0, acc_drop / 0.50)

    # ─── Phase 1: Degrade old trace ─────────────────────────────────────────
    if cluster_id in importance_masks:
        for name in importance_masks[cluster_id]:
            importance_masks[cluster_id][name] *= (1.0 - strength)
    if cluster_id in theta_star:
        current_snapshot = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
        }
        for name in theta_star[cluster_id]:
            if name in current_snapshot:
                theta_star[cluster_id][name] = (
                    theta_star[cluster_id][name] * (1.0 - strength)
                    + current_snapshot[name] * strength
                )

    # ─── Phase 2: Fine-tune on current data (no old protection) ─────────────
    old_other_loss = 0.0
    other_count = 0
    for oid in buffer.entries:
        if oid == cluster_id or not buffer.entries[oid].committed:
            continue
        oe = buffer.entries[oid]
        if not oe.core_inputs:
            continue
        old_other_loss += _eval_loss(
            model, torch.cat(oe.core_inputs, dim=0),
            torch.cat(oe.core_labels, dim=0), device
        )
        other_count += 1
    old_other_loss /= max(other_count, 1)

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

    new_other_loss = 0.0
    for oid in buffer.entries:
        if oid == cluster_id or not buffer.entries[oid].committed:
            continue
        oe = buffer.entries[oid]
        if not oe.core_inputs:
            continue
        new_other_loss += _eval_loss(
            shadow, torch.cat(oe.core_inputs, dim=0),
            torch.cat(oe.core_labels, dim=0), device
        )
        other_count += 1
    new_other_loss /= max(other_count, 1)

    forgetting = new_other_loss - old_other_loss
    if forgetting > eps_forget:
        return False

    # ─── Phase 3: Restabilize ───────────────────────────────────────────────
    model.load_state_dict(shadow.state_dict())

    # New core-set from current data
    n = min(buffer.core_size_per_cluster, current_x.size(0))
    indices = torch.randperm(current_x.size(0))[:n]
    entry.core_inputs = [current_x[indices].cpu()]
    entry.core_labels = [current_y[indices].cpu()]

    # Recompute commit accuracy
    with torch.no_grad():
        model.eval()
        preds = model(current_x[indices].to(device)).argmax(dim=1)
        entry.commit_accuracy = (preds == current_y[indices].to(device)).float().mean().item()

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
