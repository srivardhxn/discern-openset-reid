"""
Identity Prototype representation.
Stores the mean embedding and up to K diverse exemplars per enrolled identity.
Provides robust multi-instance similarity scoring.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any


class IdentityPrototype:
    """
    Compact prototype representation for an enrolled person identity.
    Combines:
    1. L2-normalized mean centroid embedding
    2. Up to K diverse exemplar embeddings
    3. Per-identity adaptive acceptance threshold tau_i
    """
    def __init__(
        self,
        identity_id: int,
        name: str,
        embeddings: np.ndarray,
        image_paths: Optional[List[str]] = None,
        max_exemplars: int = 4,
    ):
        self.identity_id = identity_id
        self.name = name
        self.raw_embeddings = np.atleast_2d(embeddings).astype(np.float32)
        self.image_paths = list(image_paths) if image_paths else []
        self.max_exemplars = max_exemplars

        # Calculate L2-normalized mean centroid
        raw_mean = np.mean(self.raw_embeddings, axis=0)
        norm = np.linalg.norm(raw_mean) + 1e-7
        self.mean_embedding: np.ndarray = (raw_mean / norm).astype(np.float32)

        # Select up to K exemplars using diversity sampling (furthest point)
        self.exemplars: np.ndarray = self._select_exemplars(self.raw_embeddings, max_exemplars)

        # Per-identity adaptive threshold (default reasonable value before calibration)
        self.adaptive_tau: float = 0.65
        self.nearest_lookalike_id: Optional[int] = None
        self.nearest_lookalike_sim: float = 0.0

    def _select_exemplars(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        """Selects up to K diverse exemplars."""
        n = len(embeddings)
        if n <= k:
            return embeddings

        # Normalize rows
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-7
        unit_embs = embeddings / norms

        selected_indices = [0]
        for _ in range(1, k):
            # Compute distance to closest selected exemplar for every point
            dist_to_selected = []
            for i in range(n):
                sims = np.dot(unit_embs[selected_indices], unit_embs[i])
                min_dist = 1.0 - np.max(sims)
                dist_to_selected.append(min_dist)

            next_idx = int(np.argmax(dist_to_selected))
            if next_idx not in selected_indices:
                selected_indices.append(next_idx)

        return embeddings[selected_indices]

    def similarity(self, query_emb: np.ndarray, use_exemplars: bool = True) -> float:
        """
        Computes cosine similarity between query and this identity prototype.
        Combines centroid similarity with best exemplar similarity.
        """
        # Centroid similarity
        centroid_sim = float(np.dot(query_emb, self.mean_embedding))

        if not use_exemplars or len(self.exemplars) == 0:
            return centroid_sim

        # Exemplar similarities
        ex_sims = np.dot(self.exemplars, query_emb)
        max_ex_sim = float(np.max(ex_sims))

        # Fused prototype similarity: 65% centroid + 35% best exemplar
        fused = 0.65 * centroid_sim + 0.35 * max_ex_sim
        return float(np.clip(fused, -1.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "name": self.name,
            "num_samples": len(self.raw_embeddings),
            "num_exemplars": len(self.exemplars),
            "adaptive_tau": round(float(self.adaptive_tau), 4),
            "nearest_lookalike_id": self.nearest_lookalike_id,
            "nearest_lookalike_sim": round(float(self.nearest_lookalike_sim), 4),
            "image_paths": self.image_paths,
        }
