"""
Curator for LOW-VARIANCE person re-identification subsets.
Computes HSV clothing histograms (upper and lower body), clusters identities,
and extracts the tightest look-alike groups plus top 10% most similar cross-identity pairs.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from .dataset import ReIDSample

def extract_hsv_color_histogram(image_path: str) -> np.ndarray:
    """
    Extracts upper and lower body HSV color histograms from person image.
    Upper body: y in [15%, 55%], x in [15%, 85%]
    Lower body: y in [55%, 90%], x in [15%, 85%]
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return np.zeros(64, dtype=np.float32)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape

    # Crop upper body (torso)
    y1_u, y2_u = int(0.15 * h), int(0.55 * h)
    x1, x2 = int(0.15 * w), int(0.85 * w)
    upper_crop = hsv[y1_u:y2_u, x1:x2]

    # Crop lower body (legs)
    y1_l, y2_l = int(0.55 * h), int(0.90 * h)
    lower_crop = hsv[y1_l:y2_l, x1:x2]

    # Compute 16-bin H, 8-bin S, 8-bin V for upper and lower
    def get_hist(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(32, dtype=np.float32)
        h_hist = cv2.calcHist([crop], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([crop], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([crop], [2], None, [8], [0, 256]).flatten()
        combined = np.concatenate([h_hist, s_hist, v_hist])
        norm = np.linalg.norm(combined) + 1e-7
        return combined / norm

    upper_hist = get_hist(upper_crop)
    lower_hist = get_hist(lower_crop)
    full_hist = np.concatenate([upper_hist, lower_hist])
    full_norm = np.linalg.norm(full_hist) + 1e-7
    return (full_hist / full_norm).astype(np.float32)


def curate_low_variance_subset(
    samples: List[ReIDSample],
    top_pairs_ratio: float = 0.10,
    min_cluster_size: int = 2,
    output_path: str = "results/lowvar_subset.json",
) -> Dict[str, Any]:
    """
    Analyzes clothing histograms across identities, clusters them into appearance groups,
    and identifies:
    1. Tightest look-alike groups (clusters with lowest internal variance)
    2. Top 10% most similar cross-identity pairs
    Saves metadata to output_path.
    """
    # Group samples by identity
    id_to_samples: Dict[int, List[ReIDSample]] = {}
    for s in samples:
        id_to_samples.setdefault(s.identity_id, []).append(s)

    identities = sorted(list(id_to_samples.keys()))
    if len(identities) < 2:
        return {"error": "Need at least 2 identities to curate low-variance subset"}

    # Extract mean color profile per identity
    id_histograms: Dict[int, np.ndarray] = {}
    for pid in identities:
        hists = [extract_hsv_color_histogram(s.image_path) for s in id_to_samples[pid]]
        mean_hist = np.mean(hists, axis=0)
        mean_hist = mean_hist / (np.linalg.norm(mean_hist) + 1e-7)
        id_histograms[pid] = mean_hist

    hist_matrix = np.stack([id_histograms[pid] for pid in identities])
    sim_matrix = cosine_similarity(hist_matrix)

    # Hierarchical clustering of identities based on clothing histogram distance
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    # Estimate reasonable cluster count
    n_clusters = max(2, min(len(identities) // 3, 8))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters: Dict[int, List[int]] = {}
    for idx, pid in enumerate(identities):
        c_label = int(labels[idx])
        clusters.setdefault(c_label, []).append(pid)

    # Compute intra-cluster average similarity for each cluster
    cluster_stats = []
    for c_id, member_pids in clusters.items():
        if len(member_pids) < min_cluster_size:
            continue
        member_indices = [identities.index(p) for p in member_pids]
        sub_sim = sim_matrix[np.ix_(member_indices, member_indices)]
        # Average off-diagonal similarity
        triu_indices = np.triu_indices(len(member_pids), k=1)
        if len(triu_indices[0]) > 0:
            avg_sim = float(np.mean(sub_sim[triu_indices]))
        else:
            avg_sim = 1.0

        cluster_stats.append({
            "cluster_id": c_id,
            "identities": member_pids,
            "size": len(member_pids),
            "intra_cluster_similarity": round(avg_sim, 4),
        })

    # Sort clusters by tightest similarity (descending similarity)
    cluster_stats.sort(key=lambda x: x["intra_cluster_similarity"], reverse=True)

    # Compute all cross-identity pairs and extract top top_pairs_ratio (e.g. 10%)
    cross_pairs = []
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            pid_i = identities[i]
            pid_j = identities[j]
            sim = float(sim_matrix[i, j])
            cross_pairs.append({
                "identity_a": pid_i,
                "identity_b": pid_j,
                "clothing_similarity": round(sim, 4),
                "is_same_cluster": bool(labels[i] == labels[j]),
            })

    cross_pairs.sort(key=lambda x: x["clothing_similarity"], reverse=True)
    num_top_pairs = max(1, int(len(cross_pairs) * top_pairs_ratio))
    top_lookalike_pairs = cross_pairs[:num_top_pairs]

    # Select low-variance subset identities:
    # Union of identities in top tightest clusters and top 10% cross pairs
    subset_ids_set = set()
    # Add top 50% tightest clusters
    top_clusters = cluster_stats[:max(1, len(cluster_stats) // 2)]
    for c in top_clusters:
        subset_ids_set.update(c["identities"])

    for p in top_lookalike_pairs:
        subset_ids_set.add(p["identity_a"])
        subset_ids_set.add(p["identity_b"])

    subset_identities = sorted(list(subset_ids_set))

    # Collect sample image paths for low-variance subset
    subset_sample_paths = []
    for pid in subset_identities:
        for s in id_to_samples[pid]:
            subset_sample_paths.append(s.image_path)

    result_data = {
        "metadata": {
            "total_identities": len(identities),
            "low_variance_identities_count": len(subset_identities),
            "total_samples": sum(len(v) for v in id_to_samples.values()),
            "low_variance_samples_count": len(subset_sample_paths),
            "top_pairs_ratio": top_pairs_ratio,
            "num_clusters": len(clusters),
        },
        "tightest_clusters": cluster_stats,
        "top_lookalike_pairs": top_lookalike_pairs,
        "low_variance_identity_ids": subset_identities,
        "low_variance_sample_paths": subset_sample_paths,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return result_data
