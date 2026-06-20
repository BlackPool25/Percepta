"""
Per-layer stratified freeze experiment.

Selects top-K% of parameters by Fisher within EACH layer independently,
guaranteeing that fc2's highest-Fisher neurons and fc1's highest-Fisher
neurons are included in the frozen set.
"""
import torch
import torch.nn as nn
import numpy as np

from model import create_model
from data import get_split_mnist_tasks
from metrics import evaluate
from baselines import compute_fisher_diag, normalize_fisher_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── 1. Train task 0 ─────────────────────────────────────────────────────────

model = create_model().to(device)
tasks = get_split_mnist_tasks(batch_size=128)
train_0, test_0 = tasks[0]

criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
for epoch in range(2):
    for x, y in train_0:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); criterion(model(x), y).backward(); opt.step()

base_acc = evaluate(model, test_0, device)
print(f'Base task 0 acc: {base_acc:.4f}')

# ─── 2. Compute and normalize Fisher ─────────────────────────────────────────

f0_raw = compute_fisher_diag(model, train_0, device)
f0 = normalize_fisher_mask(f0_raw)

theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}

# ─── 3. Per-layer parameter grouping for selection ───────────────────────────

def get_param_groups_by_layer(model):
    layers = {}
    for name, param in model.named_parameters():
        layer = name.rsplit('.', 1)[0]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append((name, param.numel()))
    return layers

layer_info = get_param_groups_by_layer(model)
print(f'\nLayers:')
for layer, params in layer_info.items():
    total = sum(n for _, n in params)
    print(f'  {layer}: {len(params)} params, {total} values')

# ─── 4. Per-layer top-K% selection ───────────────────────────────────────────

freeze_masks = {}  # param_name -> boolean mask

for pct in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]:
    freeze_masks[pct] = {}
    for name, param in model.named_parameters():
        freeze_masks[pct][name] = torch.zeros_like(param, dtype=torch.bool)

    for layer_name in layer_info:
        layer_params = [n for n, _ in layer_info[layer_name]]
        # Collect all Fisher values for this layer
        layer_fisher = torch.cat([f0[n].flatten() for n in layer_params])
        k = max(1, int(layer_fisher.numel() * pct / 100.0))
        _, top_indices = torch.topk(layer_fisher, k)
        
        # Map back to per-parameter masks
        offset = 0
        for name in layer_params:
            n_vals = model.state_dict()[name].numel()
            param_mask = torch.zeros(n_vals, dtype=torch.bool)
            for idx in top_indices:
                if offset <= idx < offset + n_vals:
                    param_mask[idx - offset] = True
            freeze_masks[pct][name] = param_mask.view(f0[name].shape)
            offset += n_vals
    
    total_frozen = sum(freeze_masks[pct][n].sum().item() for n in freeze_masks[pct])
    print(f'\npct={pct}%: {total_frozen} params frozen ({100*total_frozen/sum(p.numel() for p in model.parameters()):.2f}% of total)')

# ─── 5. Freeze sweep: train task 1 with protected params ─────────────────────

train_1, test_1 = tasks[1]
results = []

for pct in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]:
    print(f'\n--- pct={pct}% per layer ---')
    
    model2 = create_model().to(device)
    model2.load_state_dict(theta0)
    
    # Create optimizer with per-parameter learning rates (0 for frozen)
    param_groups = []
    for name, param in model2.named_parameters():
        mask = freeze_masks[pct].get(name)
        if mask is not None and mask.any():
            # This param has some frozen + some unfrozen elements
            # SGD can't do per-element LR, so use same LR for all
            # We'll enforce freezing by resetting after each step
            pass
    
    opt = torch.optim.SGD(model2.parameters(), lr=1e-3, momentum=0.9)
    
    for epoch in range(2):
        for x, y in train_1:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model2(x), y)
            loss.backward()
            opt.step()
            
            # Enforce freezing: restore frozen params to theta0 values
            for name in freeze_masks[pct]:
                mask = freeze_masks[pct][name]
                if mask.any():
                    state = model2.state_dict()
                    state[name][mask] = theta0[name][mask].to(state[name].device)
                    model2.load_state_dict(state)
    
    t0_acc = evaluate(model2, test_0, device)
    t1_acc = evaluate(model2, test_1, device)
    
    # Measure how much each layer's frozen vs unfrozen params changed
    layer_changes = {}
    for layer_name in layer_info:
        layer_params = [n for n, _ in layer_info[layer_name]]
        frozen_change = 0.0
        frozen_count = 0
        free_change = 0.0
        free_count = 0
        for name in layer_params:
            mask = freeze_masks[pct][name]
            delta = (model2.state_dict()[name] - theta0[name].to(device)).pow(2)
            if mask.any():
                frozen_change += delta[mask].sum().item()
                frozen_count += mask.sum().item()
            if (~mask).any():
                free_change += delta[~mask].sum().item()
                free_count += (~mask).sum().item()
        layer_changes[layer_name] = {
            'frozen': frozen_change / max(frozen_count, 1),
            'free': free_change / max(free_count, 1),
        }
    
    results.append({
        'pct': pct,
        't0_acc': t0_acc,
        't1_acc': t1_acc,
        'layer_changes': layer_changes,
    })
    print(f'  t0={t0_acc:.4f}, t1={t1_acc:.4f}')
    for ln, lc in layer_changes.items():
        print(f'    {ln}: frozen_mse={lc["frozen"]:.8f}, free_mse={lc["free"]:.8f}')

# ─── 6. Summary ──────────────────────────────────────────────────────────────

print('\n' + '=' * 60)
print('PER-LAYER FREEZE SUMMARY')
print('=' * 60)
print(f'{"pct":>6} {"t0_acc":>8} {"t1_acc":>8}  note')
print('-' * 40)
for r in results:
    note = ''
    if r['t0_acc'] > 0.5:
        note = '← substantial retention'
    elif r['t0_acc'] > 0.1:
        note = '← partial retention'
    print(f'{r["pct"]:>6.1f} {r["t0_acc"]:>8.4f} {r["t1_acc"]:>8.4f}  {note}')

print()
if results[-1]['t0_acc'] < base_acc * 0.1:
    print('CONCLUSION: Even with per-layer stratified freezing including fc2,')
    print('accuracy collapses. Supports the distributed-collapse hypothesis.')
    print('Fisher masking cannot isolate task-0 performance via per-parameter protection.')
elif results[-1]['t0_acc'] > base_acc * 0.5:
    print('CONCLUSION: Accuracy substantially recovers with per-layer freezing.')
    print('The original result was selection bias toward early layers,')
    print('not a structural Fisher limitation. Layer-stratified protection works.')
else:
    print('CONCLUSION: Partial recovery. Some protection possible via')
    print('layer-stratified freezing but distributed interactions limit it.')
