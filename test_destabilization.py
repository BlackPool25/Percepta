"""
Test: recall-destabilization under synthetic concept drift.

1. Train model on Split-MNIST tasks 0-4 (digits 0-9)
2. Introduce drift: Task 5 swaps digit 0↔1 labels
3. Verify destabilization detects the drift and restabilizes
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from model import create_model
from data import get_split_mnist_tasks, TASK_CLASSES
from metrics import evaluate
from buffer import EpisodicBuffer
from gate import (
    is_candidate, validate_and_commit,
    check_destabilization, destabilize_and_restabilize,
)
from train import build_replay_sample, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

SEED = 42
set_seed(SEED)

# ─── 1. Train gate model on Split-MNIST tasks 0-4 ────────────────────────────

model = create_model(k_wta=64, use_fast_layer=True, fast_hidden=64, fast_lr=0.01).to(device)
tasks = get_split_mnist_tasks(batch_size=128)

buffer = EpisodicBuffer(max_size=10000, core_size_per_cluster=100)
fisher_masks = {}
theta_star = {}
committed_clusters = set()

lr = 1e-3
fast_lr = 0.01
lambda_ewc = 0.1
momentum = 0.9
freq_threshold = 500
persist_window = 10
eps_gain = 0.001
eps_forget = 0.50
stage2_fine_tune_steps = 5
stage2_fine_tune_lr = 1e-4
replay_sample_size = 512

for task_id, (train_loader, test_loader) in enumerate(tasks):
    print(f'\n=== Task {task_id} (digits {TASK_CLASSES[task_id]}) ===')
    merged_f, merged_t = None, None
    has_fast = hasattr(model, 'get_fast_params') and len(model.get_fast_params()) > 0
    slow_optimizer = torch.optim.SGD(model.get_slow_params(), lr=lr, momentum=momentum)
    if has_fast:
        fast_optimizer = torch.optim.SGD(model.get_fast_params(), lr=fast_lr)

    replay_train_loader = None
    if committed_clusters:
        replay_train_loader = build_replay_sample(committed_clusters, buffer, replay_sample_size, device)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1):
        model.train()
        replay_iter = iter(replay_train_loader) if replay_train_loader else None
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            slow_optimizer.zero_grad()
            if has_fast:
                fast_optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)

            if replay_iter is not None:
                try:
                    rx, ry = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_train_loader)
                    rx, ry = next(replay_iter)
                rx, ry = rx.to(device), ry.to(device)
                loss = loss + criterion(model(rx), ry)

            loss.backward()
            slow_optimizer.step()
            if has_fast:
                fast_optimizer.step()

            for class_id in batch_y.unique().tolist():
                mask = batch_y == class_id
                buffer.add(cluster_id=class_id, inputs=batch_x[mask].cpu(), labels=batch_y[mask].cpu())

        # Promotion gate
        buffer.recompute_errors(model, device)
        for entry in buffer.get_candidates():
            if is_candidate(entry, freq_threshold, persist_window):
                if entry.cluster_id in committed_clusters:
                    continue
                rl = build_replay_sample(committed_clusters, buffer, replay_sample_size, device)
                validated = validate_and_commit(model, entry, fisher_masks, theta_star,
                                                eps_gain, eps_forget, rl,
                                                stage2_fine_tune_lr, stage2_fine_tune_steps,
                                                lambda_ewc, device)
                if validated:
                    committed_clusters.add(entry.cluster_id)
                    buffer.commit_cluster(entry.cluster_id)
                    replay_train_loader = build_replay_sample(committed_clusters, buffer, replay_sample_size, device)
                    replay_iter = iter(replay_train_loader) if replay_train_loader else None

        if has_fast:
            model.reset_fast_weights()

    acc = evaluate(model, test_loader, device)
    print(f'  Task {task_id} test acc: {acc:.4f}')
    print(f'  Committed clusters: {sorted(committed_clusters)}')

# ─── 2. Record baseline test accuracies ──────────────────────────────────────

print('\n=== Baseline accuracies (before drift) ===')
baseline_accs = []
for t in range(5):
    acc = evaluate(model, tasks[t][1], device)
    baseline_accs.append(acc)
    print(f'  Task {t} ({TASK_CLASSES[t]}): {acc:.4f}')

# ─── 3. Introduce drift: swap labels for digit 0 and 1 ──────────────────────

print('\n=== Introducing DRIFT: swapping labels for digits 0 and 1 ===')
"""
We create a drift task where all images of digit 0 get label 1,
and all images of digit 1 get label 0. We add this data to the
buffer under cluster_id=0 and cluster_id=1, with SWAPPED labels.

This simulates concept drift: the model's committed belief about
"this is digit 0" is now wrong.
"""
drift_train = tasks[0][0]  # Same data loader as task 0 (digits 0,1)
drift_dataset = drift_train.dataset

drift_inputs_0, drift_labels_0 = [], []
drift_inputs_1, drift_labels_1 = [], []

for i in range(len(drift_dataset)):
    img, label = drift_dataset[i]
    if label == 0:
        drift_inputs_0.append(img.unsqueeze(0))
        drift_labels_0.append(torch.tensor([1]))  # SWAPPED: 0→1
    elif label == 1:
        drift_inputs_1.append(img.unsqueeze(0))
        drift_labels_1.append(torch.tensor([0]))  # SWAPPED: 1→0

print(f'  Drift samples for class 0 (now labeled 1): {len(drift_inputs_0)}')
print(f'  Drift samples for class 1 (now labeled 0): {len(drift_inputs_1)}')

# Add drift data to buffer (append to existing clusters with new labels)
for inputs, labels in [(drift_inputs_0, drift_labels_0), (drift_inputs_1, drift_labels_1)]:
    for inp, lbl in zip(inputs, labels):
        cid = lbl.item()  # The DRIFT label is the cluster
        buffer.add(cluster_id=cid, inputs=inp.cpu(), labels=lbl.cpu())

print('\n=== Accuracies AFTER drift buffer fill (before destabilization) ===')
for t in range(5):
    acc = evaluate(model, tasks[t][1], device)
    print(f'  Task {t} ({TASK_CLASSES[t]}): {acc:.4f}')

# ─── 4. Run destabilization ──────────────────────────────────────────────────

print('\n=== Running destabilization check ===')
destab_candidates = check_destabilization(
    model, buffer, committed_clusters, device,
    destabilize_threshold=0.80,
    destabilize_cooldown=0,
)
print(f'  Destabilization candidates: {destab_candidates}')

for cid in destab_candidates:
    print(f'\n  Destabilizing cluster {cid}...')
    success = destabilize_and_restabilize(
        model, buffer, cid, device,
        steps=40, lr=1e-4, eps_forget=0.50,
    )
    print(f'  Cluster {cid} destabilization: {"SUCCESS" if success else "FAILED"}')

# ─── 5. Evaluate post-destabilization ────────────────────────────────────────

print('\n=== Accuracies AFTER destabilization ===')
for t in range(5):
    acc = evaluate(model, tasks[t][1], device)
    print(f'  Task {t} ({TASK_CLASSES[t]}): {acc:.4f}')

# Check: task 0 accuracy should be HIGH (model learned the new mapping)
# But the test loader still has original labels, so task 0 test accuracy
# will appear LOW if model accepts the drift. That's EXPECTED behavior.
# The correct test is: does the model correctly classify digit 0 as class 1 now?

print('\n=== Validating drift acceptance ===')
# If destabilization worked, the model should classify digit 0 as class 1
# and digit 1 as class 0 (the new mapping)

drift_inputs = torch.cat(drift_inputs_0[:50] + drift_inputs_1[:50], dim=0)
drift_labels = torch.cat(drift_labels_0[:50] + drift_labels_1[:50], dim=0)

model.eval()
with torch.no_grad():
    outputs = model(drift_inputs.to(device))
    preds = outputs.argmax(dim=1).cpu()
    drift_acc = (preds == drift_labels).float().mean().item()

print(f'  Accuracy on DRIFT data (0→1, 1→0): {drift_acc:.4f}')

# Check collateral damage on tasks 2-4
other_accs = []
for t in range(2, 5):
    acc = evaluate(model, tasks[t][1], device)
    other_accs.append(acc)
    print(f'  Task {t} ({TASK_CLASSES[t]}): {acc:.4f}')

print(f'\n  Before drift: Task 0-4 avg = {sum(baseline_accs)/5:.4f}')
print(f'  After drift:  Other tasks (2-4) avg = {sum(other_accs)/3:.4f}')

if drift_acc > 0.70:
    print('\n✓ Destabilization: SUCCESS — model adapted to drift')
else:
    print('\n✗ Destabilization: PARTIAL — drift not fully absorbed')

if min(other_accs) > baseline_accs[2] * 0.5:
    print('✓ Collateral damage: ACCEPTABLE — other tasks mostly preserved')
else:
    print('✗ Collateral damage: SEVERE — destabilization damaged other tasks')
