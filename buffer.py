import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class BufferEntry:
    cluster_id: int
    inputs: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)
    logits: list[torch.Tensor] = field(default_factory=list)
    features: list[torch.Tensor] = field(default_factory=list)
    core_inputs: list[torch.Tensor] = field(default_factory=list)
    core_labels: list[torch.Tensor] = field(default_factory=list)
    core_logits: list[torch.Tensor] = field(default_factory=list)
    core_features: list[torch.Tensor] = field(default_factory=list)
    pred_error_history: list[float] = field(default_factory=list)
    core_acc_history: list[float] = field(default_factory=list)
    seen_count: int = 0
    committed: bool = False
    destabilize_count: int = 0
    last_destabilized_step: int = 0
    commit_accuracy: float = 0.0


class EpisodicBuffer:
    def __init__(self, max_size: int, core_size_per_cluster: int = 0):
        self.max_size = max_size
        self.core_size_per_cluster = core_size_per_cluster
        self.entries: dict[int, BufferEntry] = {}

    def add(self, cluster_id: int, inputs: torch.Tensor, labels: torch.Tensor,
            logits: torch.Tensor | None = None):
        if cluster_id not in self.entries:
            self.entries[cluster_id] = BufferEntry(cluster_id=cluster_id)

        entry = self.entries[cluster_id]
        entry.inputs.append(inputs.cpu())
        entry.labels.append(labels.cpu())
        if logits is not None:
            entry.logits.append(logits.cpu())
        entry.seen_count += inputs.size(0)

        self._evict_if_needed(cluster_id)

    def _evict_if_needed(self, cluster_id: int):
        entry = self.entries[cluster_id]
        core_total = sum(t.size(0) for t in entry.core_inputs) if entry.committed else 0
        total_samples = sum(t.size(0) for t in entry.inputs) + core_total
        if total_samples > self.max_size:
            # FIFO eviction: remove oldest blocks, but never touch core-set
            while total_samples > self.max_size and entry.inputs:
                removed = entry.inputs.pop(0)
                total_samples -= removed.size(0)
                entry.labels.pop(0)

    def commit_cluster(self, cluster_id: int, model=None, device=None):
        entry = self.entries.get(cluster_id)
        if entry is None or entry.committed:
            return
        entry.committed = True
        if self.core_size_per_cluster <= 0 or not entry.inputs:
            entry.core_inputs = [torch.cat(entry.inputs, dim=0)]
            entry.core_labels = [torch.cat(entry.labels, dim=0)]
            if entry.logits:
                entry.core_logits = [torch.cat(entry.logits, dim=0)]
            return
        all_inputs = torch.cat(entry.inputs, dim=0)
        all_labels = torch.cat(entry.labels, dim=0)
        n = min(self.core_size_per_cluster, all_inputs.size(0))
        indices = torch.linspace(0, all_inputs.size(0) - 1, n).long()
        entry.core_inputs = [all_inputs[indices]]
        entry.core_labels = [all_labels[indices]]
        if entry.logits:
            all_logits = torch.cat(entry.logits, dim=0)
            entry.core_logits = [all_logits[indices]]

        # Store commit-time accuracy for relative drift detection
        if model is not None and device is not None:
            with torch.no_grad():
                model.eval()
                core_x = torch.cat(entry.core_inputs, dim=0).to(device)
                core_y = torch.cat(entry.core_labels, dim=0).to(device)
                preds = model(core_x).argmax(dim=1)
                entry.commit_accuracy = (preds == core_y).float().mean().item()

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

    @torch.no_grad()
    def cache_features(self, feature_fn, device):
        """Pre-compute fc1 features for all buffer entries using feature_fn.

        feature_fn takes a batch of inputs and returns feature vectors.
        Call after recompute_errors or when adding new data.
        """
        for entry in self.entries.values():
            if not entry.inputs:
                continue
            all_x = torch.cat(entry.inputs, dim=0)
            feats = feature_fn(all_x.to(device)).cpu()
            entry.features = [feats]
            if entry.core_inputs:
                core_x = torch.cat(entry.core_inputs, dim=0)
                core_feats = feature_fn(core_x.to(device)).cpu()
                entry.core_features = [core_feats]

    @torch.no_grad()
    def retrieve(self, query_feat: torch.Tensor, k: int = 5,
                 use_core: bool = True,
                 device=None) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Retrieve top-k most similar stored examples by cosine similarity.

        Returns list of (retrieved_inputs, retrieved_labels, retrieved_features) tuples.
        If device is specified, returned tensors are moved to that device.
        """
        q = query_feat.cpu()
        q_norm = q / (q.norm() + 1e-8)

        candidates = []
        for entry in self.entries.values():
            src = entry.core_features if use_core and entry.core_features else entry.features
            if not src:
                continue
            feats = torch.cat(src, dim=0)
            if feats.size(0) == 0:
                continue
            feats_norm = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
            sims = feats_norm @ q_norm
            val, idx = sims.topk(min(k, sims.size(0)))
            candidates.append((val, idx, entry))

        if not candidates:
            return []

        all_vals = torch.cat([c[0] for c in candidates])
        _, global_idx = all_vals.topk(min(k, all_vals.size(0)))

        results = []
        offset = 0
        for c in candidates:
            n = c[0].size(0)
            for gi in global_idx:
                if offset <= gi < offset + n:
                    local_i = gi.item() - offset
                    entry = c[2]
                    use_core_src = use_core and entry.core_features
                    src_tensor = torch.cat(entry.core_inputs if use_core_src else entry.inputs, dim=0)
                    lbl_tensor = torch.cat(entry.core_labels if use_core_src else entry.labels, dim=0)
                    feat_tensor = torch.cat(entry.core_features if use_core_src else entry.features, dim=0)
                    r_in = src_tensor[local_i].unsqueeze(0)
                    r_lbl = lbl_tensor[local_i].unsqueeze(0)
                    r_feat = feat_tensor[local_i].unsqueeze(0)
                    if device is not None:
                        r_in = r_in.to(device)
                        r_lbl = r_lbl.to(device)
                        r_feat = r_feat.to(device)
                    results.append((r_in, r_lbl, r_feat))
            offset += n

        return results[:k]

    def remove_entry(self, cluster_id: int):
        self.entries.pop(cluster_id, None)

    def __len__(self) -> int:
        return len(self.entries)
