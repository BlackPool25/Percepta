"""
Diagnostic: rank-correlation between Fisher diagonal magnitude
and empirical importance (accuracy drop from zeroing parameter groups).

This tells us whether Fisher is:
  (a) directionally correct but poorly scaled → fixable via normalization
  (b) structurally uncorrelated → diagonal Fisher is wrong instrument at this scale
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, pearsonr

from model import create_model
from data import get_split_mnist_tasks
from metrics import evaluate
from baselines import compute_fisher_diag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── 1. Train on task 0 ──────────────────────────────────────────────────────

model = create_model().to(device)
tasks = get_split_mnist_tasks(batch_size=128)
train_0, test_0 = tasks[0]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(2):
    model.train()
    for x, y in train_0:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

base_acc = evaluate(model, test_0, device)
print(f'Task 0 test accuracy: {base_acc:.4f}')

# ─── 2. Compute Fisher diagonal ──────────────────────────────────────────────

print('Computing Fisher diagonal...')
fisher = compute_fisher_diag(model, train_0, device)
print(f'  Fisher mean: {sum(f.mean().item() for f in fisher.values()) / len(fisher):.10f}')
print(f'  Fisher max:  {max(f.max().item() for f in fisher.values()):.10f}')

# ─── 3. Define parameter groups ──────────────────────────────────────────────

def get_param_groups(model: nn.Module) -> list[dict]:
    groups = []
    state = model.state_dict()

    # Conv1: 32 filters, each 1x3x3 kernel + bias
    conv1_w = state['conv1.weight']  # (32, 1, 3, 3)
    conv1_b = state['conv1.bias']    # (32,)
    for i in range(conv1_w.shape[0]):
        groups.append({
            'name': f'conv1_filter_{i}',
            'params': {
                'conv1.weight': (slice(i, i+1), ...),
                'conv1.bias': (i,),
            },
            'fisher_mask': {'conv1.weight': fisher['conv1.weight'][i:i+1],
                            'conv1.bias': fisher['conv1.bias'][i]},
        })

    # Conv2: 64 filters, each 32x3x3 kernel + bias
    conv2_w = state['conv2.weight']  # (64, 32, 3, 3)
    conv2_b = state['conv2.bias']    # (64,)
    for i in range(conv2_w.shape[0]):
        groups.append({
            'name': f'conv2_filter_{i}',
            'params': {
                'conv2.weight': (slice(i, i+1), ...),
                'conv2.bias': (i,),
            },
            'fisher_mask': {'conv2.weight': fisher['conv2.weight'][i:i+1],
                            'conv2.bias': fisher['conv2.bias'][i]},
        })

    # FC1: 256 output neurons, each 12544 weights + 1 bias
    fc1_w = state['fc1.weight']  # (256, 12544)
    fc1_b = state['fc1.bias']    # (256,)
    for i in range(fc1_w.shape[0]):
        groups.append({
            'name': f'fc1_neuron_{i}',
            'params': {
                'fc1.weight': (i, ...),
                'fc1.bias': (i,),
            },
            'fisher_mask': {'fc1.weight': fisher['fc1.weight'][i],
                            'fc1.bias': fisher['fc1.bias'][i]},
        })

    # FC2: 10 output neurons, each 256 weights + 1 bias
    fc2_w = state['fc2.weight']  # (10, 256)
    fc2_b = state['fc2.bias']    # (10,)
    for i in range(fc2_w.shape[0]):
        groups.append({
            'name': f'fc2_neuron_{i}',
            'params': {
                'fc2.weight': (i, ...),
                'fc2.bias': (i,),
            },
            'fisher_mask': {'fc2.weight': fisher['fc2.weight'][i],
                            'fc2.bias': fisher['fc2.bias'][i]},
        })

    return groups

groups = get_param_groups(model)
print(f'Total parameter groups: {len(groups)}')

# ─── 4. Compute empirical importance via zeroing ──────────────────────────────

print('Measuring empirical importance (zeroing each group)...')
results = []

for g_idx, group in enumerate(groups):
    if g_idx % 50 == 0:
        print(f'  Group {g_idx}/{len(groups)}...')

    state = model.state_dict()
    original_vals = {}

    for param_name, index in group['params'].items():
        original_vals[param_name] = state[param_name][index].clone()
        # Zero out the parameter group
        state[param_name][index] = 0.0

    model.load_state_dict(state)
    acc_after = evaluate(model, test_0, device)
    importance = base_acc - acc_after

    # Compute mean Fisher magnitude for this group
    fisher_vals = []
    for param_name, idx in group['params'].items():
        if isinstance(idx, tuple):
            fisher_vals.append(fisher[param_name][idx].abs().mean().item())
        else:
            fisher_vals.append(fisher[param_name][idx].abs().mean().item())
    mean_fisher = np.mean(fisher_vals)

    # Restore
    for param_name, index in group['params'].items():
        state[param_name][index] = original_vals[param_name]
    model.load_state_dict(state)

    results.append({
        'name': group['name'],
        'importance': importance,
        'mean_fisher': mean_fisher,
    })

# ─── 5. Analyze ──────────────────────────────────────────────────────────────

importances = np.array([r['importance'] for r in results])
fisher_vals = np.array([r['mean_fisher'] for r in results])

# Remove groups with zero importance (uninteresting for correlation)
nonzero_mask = importances > 1e-8
nonzero_imp = importances[nonzero_mask]
nonzero_fish = fisher_vals[nonzero_mask]

print('\n' + '=' * 60)
print('RANK CORRELATION: FISHER vs EMPIRICAL IMPORTANCE')
print('=' * 60)

if len(nonzero_imp) > 2:
    rho, p_val = spearmanr(nonzero_fish, nonzero_imp)
    rho_all, p_all = spearmanr(fisher_vals, importances)
    pr, pp = pearsonr(np.log1p(fisher_vals), importances)

    print(f'\nAll {len(results)} groups:')
    print(f'  Spearman ρ = {rho_all:.4f}  (p={p_all:.6f})')

    print(f'\n{len(nonzero_imp)} groups with non-zero importance:')
    print(f'  Spearman ρ = {rho:.4f}  (p={p_val:.6f})')

    print(f'\nPearson r(log(Fisher), importance) = {pr:.4f}  (p={pp:.6f})')
else:
    print('Too few non-zero importance groups for correlation.')

print(f'\nFisher stats: mean={fisher_vals.mean():.10f}, std={fisher_vals.std():.10f}, '
      f'median={np.median(fisher_vals):.10f}')
print(f'Importance stats: mean={importances.mean():.6f}, std={importances.std():.6f}, '
      f'max={importances.max():.6f}')

# Top-K analysis: do the top 5% Fisher groups contain the most important groups?
top_k = max(1, len(results) // 20)
top_fisher_idx = np.argsort(fisher_vals)[-top_k:]
top_imp_idx = np.argsort(importances)[-top_k:]
overlap = len(set(top_fisher_idx) & set(top_imp_idx))
print(f'\nTop {top_k} groups by Fisher vs top {top_k} by importance:')
print(f'  Overlap: {overlap}/{top_k} ({100*overlap/top_k:.1f}%)')

# Per-layer analysis
layer_names = ['conv1', 'conv2', 'fc1', 'fc2']
for layer in layer_names:
    layer_results = [r for r in results if r['name'].startswith(layer)]
    if layer_results:
        lr_imp = np.array([r['importance'] for r in layer_results])
        lr_fish = np.array([r['mean_fisher'] for r in layer_results])
        lr_nonzero = lr_imp > 1e-8
        if lr_nonzero.sum() > 2:
            lr_rho, _ = spearmanr(lr_fish[lr_nonzero], lr_imp[lr_nonzero])
        else:
            lr_rho = float('nan')
        print(f'\n  {layer} ({len(layer_results)} groups, '
              f'{lr_nonzero.sum()} non-zero):')
        print(f'    Fisher mean={lr_fish.mean():.10f}, '
              f'Importance mean={lr_imp.mean():.6f}')
        print(f'    Spearman ρ = {lr_rho:.4f}')

# ─── 6. Summary ──────────────────────────────────────────────────────────────

print('\n' + '=' * 60)
print('INTERPRETATION')
print('=' * 60)
print("""
If ρ > 0.5 (p < 0.05):
  Fisher is directionally correct → fix via scaling/normalization suffices
  The v3 recall-destabilization plan (D5/D6) survives with rescaling

If ρ < 0.2 or p > 0.05:
  Diagonal Fisher is the wrong instrument at this scale
  The architecture needs a different importance mechanism (SI, empirical)
  This threatens the dependency chain past v1 regardless of model size

Edge case: ρ moderate but driven entirely by one layer:
  The Fisher works for some parameter types but not others
  Hybrid approach: SI for dense layers, Fisher for conv layers (or vice versa)
""")
