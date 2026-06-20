import argparse
import csv
import json
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import create_model
from data import get_split_mnist_tasks
from metrics import evaluate, compute_acc, compute_bwt
from baselines import train_naive, train_ewc, ewc_penalty, merge_fisher_masks, compute_fisher_diag
from buffer import EpisodicBuffer
from gate import is_candidate, validate_and_commit

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'epochs_per_task': 2,
    'learning_rate': 1e-3,
    'batch_size': 128,
    'lambda_ewc': 0.1,
    'freq_threshold': 50,
    'persist_window': 10,
    'eps_gain': 0.01,
    'eps_forget': 0.05,
    'stage2_fine_tune_steps': 5,
    'stage2_fine_tune_lr': 1e-4,
    'promo_gate_interval': 1,
    'buffer_max_size': 10000,
    'replay_sample_size': 512,
    'momentum': 0.9,
    'seeds': [42, 43, 44, 45, 46],
    'configs': ['naive', 'ewc', 'two_stage_gate'],
    'device': None,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_replay_sample(
    committed_clusters: set,
    buffer: EpisodicBuffer,
    sample_size: int,
    device: torch.device,
) -> DataLoader | None:
    if not committed_clusters:
        return None
    all_inputs = []
    all_labels = []
    for cid in committed_clusters:
        if cid not in buffer.entries:
            continue
        entry = buffer.entries[cid]
        if not entry.inputs:
            continue
        inputs_cat = torch.cat(entry.inputs, dim=0)
        labels_cat = torch.cat(entry.labels, dim=0)
        n = inputs_cat.size(0)
        n_sample = min(n, sample_size // len(committed_clusters))
        perm = torch.randperm(n)[:n_sample]
        all_inputs.append(inputs_cat[perm])
        all_labels.append(labels_cat[perm])
    if not all_inputs:
        return None
    ds = TensorDataset(torch.cat(all_inputs), torch.cat(all_labels))
    return DataLoader(ds, batch_size=128, shuffle=True)


# ─── Two-Stage Gate Training ──────────────────────────────────────────────────


def train_two_stage_gate(
    model: nn.Module,
    tasks: list[tuple[DataLoader, DataLoader]],
    epochs_per_task: int,
    lr: float,
    lambda_ewc: float,
    device: torch.device,
    momentum: float = 0.9,
    freq_threshold: int = 50,
    persist_window: int = 10,
    eps_gain: float = 0.01,
    eps_forget: float = 0.05,
    stage2_fine_tune_steps: int = 5,
    stage2_fine_tune_lr: float = 1e-4,
    promo_gate_interval: int = 1,
    buffer_max_size: int = 10000,
    replay_sample_size: int = 512,
) -> list[dict]:
    buffer = EpisodicBuffer(max_size=buffer_max_size)
    fisher_masks: dict[int, dict[str, torch.Tensor]] = {}
    theta_star: dict[int, dict[str, torch.Tensor]] = {}
    committed_clusters: set = set()
    task_accs_after: list[float] = []
    history = []

    # Build merged Fisher masks for ongoing EWC penalty
    def _get_merged_fisher_and_theta():
        if not fisher_masks:
            return {}, {}
        merged_f = merge_fisher_masks(list(fisher_masks.values()))
        merged_t = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        count = 0
        for ts in theta_star.values():
            for name in merged_t:
                merged_t[name] = merged_t[name] + ts[name]
            count += 1
        if count > 0:
            for name in merged_t:
                merged_t[name] /= count
        return merged_f, merged_t

    for task_id, (train_loader, test_loader) in enumerate(tasks):
        criterion = nn.CrossEntropyLoss()
        merged_f, merged_t = _get_merged_fisher_and_theta()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        for epoch in range(epochs_per_task):
            model.train()
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                if merged_f:
                    loss += ewc_penalty(model, merged_f, merged_t, lambda_ewc)
                loss.backward()
                optimizer.step()

                # Log to buffer per-sample
                for i in range(batch_x.size(0)):
                    cluster_id = batch_y[i].item()
                    buffer.add(
                        cluster_id=cluster_id,
                        inputs=batch_x[i:i+1].cpu(),
                        labels=batch_y[i:i+1].cpu(),
                    )

            # Promotion gate (every epoch)
            buffer.recompute_errors(model, device)
            for entry in buffer.get_candidates():
                if is_candidate(entry, freq_threshold, persist_window):
                    if entry.cluster_id in committed_clusters:
                        continue
                    replay_loader = build_replay_sample(
                        committed_clusters, buffer, replay_sample_size, device
                    )
                    validated = validate_and_commit(
                        model, entry, fisher_masks, theta_star,
                        eps_gain, eps_forget, replay_loader,
                        stage2_fine_tune_lr, stage2_fine_tune_steps, lambda_ewc,
                        device,
                    )
                    if validated:
                        committed_clusters.add(entry.cluster_id)

        # Record metrics after task
        acc = compute_acc(model, tasks, task_id, device)
        per_task_now = []
        for t_id in range(task_id + 1):
            per_task_now.append(evaluate(model, tasks[t_id][1], device))
        # Track accuracies right after each task is learned for BWT
        while len(task_accs_after) <= task_id:
            task_accs_after.append(0.0)
        task_accs_after[task_id] = per_task_now[-1]
        bwt = compute_bwt(per_task_now, task_accs_after[:task_id + 1])
        history.append({
            'task_id': task_id,
            'acc': acc,
            'bwt': bwt,
            'per_task_accs': per_task_now,
            'committed_clusters': sorted(list(committed_clusters)),
        })

    return history


# ─── Experiment Runner ────────────────────────────────────────────────────────


def run_experiment(
    config_name: str,
    seed: int,
    cfg: dict,
    device: torch.device,
) -> list[dict]:
    set_seed(seed)
    model = create_model().to(device)
    tasks = get_split_mnist_tasks(batch_size=cfg['batch_size'])
    history = []

    if config_name == 'naive':
        history = train_naive(model, tasks, cfg['epochs_per_task'], cfg['learning_rate'], device, cfg['momentum'])
    elif config_name == 'ewc':
        history = train_ewc(model, tasks, cfg['epochs_per_task'], cfg['learning_rate'], cfg['lambda_ewc'], device, cfg['momentum'])
    elif config_name == 'two_stage_gate':
        history = train_two_stage_gate(model, tasks, cfg['epochs_per_task'], cfg['learning_rate'], cfg['lambda_ewc'], device,
                                       momentum=cfg['momentum'], freq_threshold=cfg['freq_threshold'],
                                       persist_window=cfg['persist_window'], eps_gain=cfg['eps_gain'],
                                       eps_forget=cfg['eps_forget'], stage2_fine_tune_steps=cfg['stage2_fine_tune_steps'],
                                       stage2_fine_tune_lr=cfg['stage2_fine_tune_lr'],
                                       promo_gate_interval=cfg['promo_gate_interval'],
                                       buffer_max_size=cfg['buffer_max_size'],
                                       replay_sample_size=cfg['replay_sample_size'])
    else:
        raise ValueError(f'Unknown config: {config_name}')

    for h in history:
        h['config'] = config_name
        h['seed'] = seed

    return history


# ─── Plotting ─────────────────────────────────────────────────────────────────


def plot_results(all_results: list[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    configs = sorted(set(r['config'] for r in all_results))
    n_tasks = max(r['task_id'] for r in all_results) + 1

    for metric_name in ['acc', 'bwt']:
        plt.figure(figsize=(10, 6))
        for config in configs:
            config_results = [r for r in all_results if r['config'] == config]
            seeds = sorted(set(r['seed'] for r in config_results))
            task_vals = {t: [] for t in range(n_tasks)}
            for r in config_results:
                task_vals[r['task_id']].append(r[metric_name])
            means = [np.mean(task_vals[t]) for t in range(n_tasks)]
            stds = [np.std(task_vals[t]) for t in range(n_tasks)]
            xs = list(range(n_tasks))
            plt.plot(xs, means, marker='o', label=config)
            plt.fill_between(xs, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)

        plt.xlabel('Task ID')
        plt.ylabel(metric_name.upper())
        plt.title(f'{metric_name.upper()} across tasks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{metric_name}.png'), dpi=150)
        plt.close()


def plot_per_task_acc(all_results: list[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    configs = sorted(set(r['config'] for r in all_results))
    n_tasks = max(r['task_id'] for r in all_results) + 1

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for task_id in range(n_tasks):
        ax = axes[task_id]
        for config in configs:
            config_results = [r for r in all_results if r['config'] == config and r['task_id'] == task_id]
            vals = [r['per_task_accs'][task_id] for r in config_results]
            if vals:
                ax.bar(config, np.mean(vals), yerr=np.std(vals), alpha=0.7, label=config)
        ax.set_title(f'Task {task_id} accuracy')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_task_acc.png'), dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Percepta v1 — Split-MNIST continual learning')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu, cuda, or auto)')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_CONFIG['seeds'])
    parser.add_argument('--configs', type=str, nargs='+', default=DEFAULT_CONFIG['configs'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs_per_task'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--lambda-ewc', type=float, default=DEFAULT_CONFIG['lambda_ewc'])
    parser.add_argument('--freq-threshold', type=int, default=DEFAULT_CONFIG['freq_threshold'])
    parser.add_argument('--persist-window', type=int, default=DEFAULT_CONFIG['persist_window'])
    parser.add_argument('--eps-gain', type=float, default=DEFAULT_CONFIG['eps_gain'])
    parser.add_argument('--eps-forget', type=float, default=DEFAULT_CONFIG['eps_forget'])
    parser.add_argument('--stage2-steps', type=int, default=DEFAULT_CONFIG['stage2_fine_tune_steps'])
    parser.add_argument('--stage2-lr', type=float, default=DEFAULT_CONFIG['stage2_fine_tune_lr'])
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg['epochs_per_task'] = args.epochs
    cfg['learning_rate'] = args.lr
    cfg['lambda_ewc'] = args.lambda_ewc
    cfg['freq_threshold'] = args.freq_threshold
    cfg['persist_window'] = args.persist_window
    cfg['eps_gain'] = args.eps_gain
    cfg['eps_forget'] = args.eps_forget
    cfg['stage2_fine_tune_steps'] = args.stage2_steps
    cfg['stage2_fine_tune_lr'] = args.stage2_lr

    if args.device:
        cfg['device'] = args.device
    if cfg['device'] is None:
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(cfg['device'])
    print(f'Using device: {device}')

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for config_name in args.configs:
        for seed in args.seeds:
            print(f'Running {config_name} seed={seed}...')
            history = run_experiment(config_name, seed, cfg, device)
            all_results.extend(history)

    # Save results to CSV
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'seed', 'task_id', 'acc', 'bwt', 'per_task_accs', 'committed_clusters'])
        for r in all_results:
            writer.writerow([
                r['config'], r['seed'], r['task_id'],
                f"{r['acc']:.6f}", f"{r['bwt']:.6f}",
                json.dumps([f"{v:.4f}" for v in r.get('per_task_accs', [])]),
                json.dumps(r.get('committed_clusters', [])),
            ])

    # Save results to JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\nResults saved to {csv_path} and {json_path}')

    # Print summary
    print('\n=== Summary ===')
    configs = sorted(set(r['config'] for r in all_results))
    n_tasks = max(r['task_id'] for r in all_results) + 1
    for config in configs:
        config_results = [r for r in all_results if r['config'] == config and r['task_id'] == n_tasks - 1]
        if config_results:
            final_accs = [r['acc'] for r in config_results]
            final_bwts = [r['bwt'] for r in config_results]
            print(f'{config}: final ACC={np.mean(final_accs):.4f}±{np.std(final_accs):.4f}, BWT={np.mean(final_bwts):.4f}±{np.std(final_bwts):.4f}')

    # Generate plots
    print('\nGenerating plots...')
    plot_results(all_results, output_dir)
    plot_per_task_acc(all_results, output_dir)
    print(f'Plots saved to {output_dir}/')
    print('\nDone!')


if __name__ == '__main__':
    main()
