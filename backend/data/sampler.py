"""
Look-alike-aware PK Sampler for PyTorch DataLoader.
Constructs batches of P identities x K samples where identities in each batch
are drawn from the same clothing appearance cluster (look-alikes).
This ensures negative pairs in mini-batch are visual twins, making batch-hard triplet mining effective.
"""

from __future__ import annotations
import json
import random
from typing import Dict, List, Iterator, Optional
import torch
from torch.utils.data import Sampler
from .dataset import ReIDSample


class LookAlikePKSampler(Sampler[List[int]]):
    """
    Look-alike aware PK Sampler.
    Batches consist of P identities with K images each.
    Identities inside a batch are chosen based on visual appearance clustering.
    """
    def __init__(
        self,
        samples: List[ReIDSample],
        num_identities_per_batch: int = 4,
        num_samples_per_identity: int = 4,
        cluster_info_path: Optional[str] = None,
        use_lookalike: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.samples = samples
        self.p = num_identities_per_batch
        self.k = num_samples_per_identity
        self.use_lookalike = use_lookalike
        self.rng = random.Random(seed)

        # Map identity_id -> list of sample indices in self.samples
        self.id_to_indices: Dict[int, List[int]] = {}
        for idx, sample in enumerate(samples):
            self.id_to_indices.setdefault(sample.identity_id, []).append(idx)

        self.identities = sorted(list(self.id_to_indices.keys()))

        # Build identity clusters for look-alike grouping
        self.clusters: List[List[int]] = []
        if use_lookalike and cluster_info_path:
            try:
                with open(cluster_info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for c in data.get("tightest_clusters", []):
                        valid_ids = [pid for pid in c.get("identities", []) if pid in self.id_to_indices]
                        if len(valid_ids) >= 2:
                            self.clusters.append(valid_ids)
            except Exception:
                pass

        # Fallback if no clusters loaded: group identities sequentially
        if not self.clusters:
            # Group into chunks of size p
            self.clusters = [self.identities[i:i + self.p] for i in range(0, len(self.identities), self.p)]

    def __len__(self) -> int:
        total_samples = sum(len(indices) for indices in self.id_to_indices.values())
        return max(1, total_samples // (self.p * self.k))

    def __iter__(self) -> Iterator[List[int]]:
        # Copy identity index pools
        id_pools = {pid: list(indices) for pid, indices in self.id_to_indices.items()}
        for pid in id_pools:
            self.rng.shuffle(id_pools[pid])

        cluster_pool = [list(c) for c in self.clusters]
        self.rng.shuffle(cluster_pool)

        num_batches = len(self)

        for _ in range(num_batches):
            batch_indices: List[int] = []

            if self.use_lookalike and cluster_pool:
                # Select a cluster as the anchor group
                chosen_cluster = self.rng.choice(cluster_pool)
                self.rng.shuffle(chosen_cluster)
                selected_ids = chosen_cluster[:self.p]

                # If cluster has fewer than p identities, top up from global pool
                if len(selected_ids) < self.p:
                    remaining_ids = [pid for pid in self.identities if pid not in selected_ids]
                    self.rng.shuffle(remaining_ids)
                    selected_ids.extend(remaining_ids[:self.p - len(selected_ids)])
            else:
                # Standard random PK sampling
                selected_ids = self.rng.sample(self.identities, min(self.p, len(self.identities)))

            for pid in selected_ids:
                pool = id_pools[pid]
                while len(pool) < self.k:
                    # Replenish with replacement if identity has fewer samples than k
                    fresh = list(self.id_to_indices[pid])
                    self.rng.shuffle(fresh)
                    if not fresh:
                        break
                    pool.extend(fresh)

                sampled = [pool.pop() for _ in range(self.k)]
                batch_indices.extend(sampled)

            yield batch_indices
