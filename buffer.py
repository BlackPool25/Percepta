import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class BufferEntry:
    cluster_id: int
    inputs: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    pred_error_history: list[float] = field(default_factory=list)
    seen_count: int = 0


class EpisodicBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.entries: dict[int, BufferEntry] = {}

    def add(self, cluster_id: int, inputs: torch.Tensor, labels: torch.Tensor):
        if cluster_id not in self.entries:
            self.entries[cluster_id] = BufferEntry(cluster_id=cluster_id)

        entry = self.entries[cluster_id]
        # We store per-sample; inputs/labels are tensors from the batch
        # For simplicity, store the whole batch as one "experience block"
        entry.inputs.append(inputs.cpu())
        entry.labels.append(labels.cpu())
        entry.seen_count += inputs.size(0)

        self._evict_if_needed(cluster_id)

    def _evict_if_needed(self, cluster_id: int):
        entry = self.entries[cluster_id]
        total_samples = sum(t.size(0) for t in entry.inputs)
        if total_samples > self.max_size:
            # FIFO eviction: remove oldest blocks
            while total_samples > self.max_size and entry.inputs:
                removed = entry.inputs.pop(0)
                total_samples -= removed.size(0)
                entry.labels.pop(0)

    @torch.no_grad()
    def recompute_errors(self, model: nn.Module, device: torch.device):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        model.eval()
        for entry in self.entries.values():
            if not entry.inputs:
                continue
            all_inputs = torch.cat(entry.inputs, dim=0)
            all_labels = torch.cat(entry.labels, dim=0)
            n = all_inputs.size(0)
            # Process in chunks to avoid OOM with large buffers
            losses = []
            chunk_size = 256
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                x = all_inputs[start:end].to(device)
                y = all_labels[start:end].to(device)
                loss = criterion(model(x), y).item()
                losses.append(loss)
            avg_loss = sum(losses) / len(losses)
            entry.pred_error_history.append(avg_loss)

    def get_candidates(self) -> list[BufferEntry]:
        return list(self.entries.values())

    def get_committed_candidates(self, committed_clusters: set) -> list[BufferEntry]:
        committed_ids = set(range(10)) & committed_clusters
        return [e for e in self.entries.values() if e.cluster_id in committed_ids]

    def remove_entry(self, cluster_id: int):
        self.entries.pop(cluster_id, None)

    def __len__(self) -> int:
        return len(self.entries)
