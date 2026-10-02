"""
Metrics and ROC / DIR evaluation engine for open-set person re-identification.
Computes:
- ROC (TAR vs FAR curve)
- TAR @ FAR = 1% (0.01) and FAR = 0.1% (0.001)
- AUROC (Area under ROC curve)
- Detection and Identification Rate (DIR) at target FAR
- Operating point analysis
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import roc_curve, auc


def compute_roc_metrics(
    genuine_scores: List[float],
    impostor_scores: List[float],
    genuine_correct_id: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """
    Computes complete ROC metrics for open-set verification/identification.
    genuine_scores: scores of positive matches
    impostor_scores: scores of impostor matches
    genuine_correct_id: boolean flag indicating if top-1 identity was correct
    """
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    y_scores = np.array(genuine_scores + impostor_scores)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "tar_at_far_1pct": 0.0,
            "tar_at_far_01pct": 0.0,
            "dir_at_far_1pct": 0.0,
            "roc_curve_points": [],
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    # Helper to find TAR at specific FAR target
    def get_tar_at_far(target_far: float) -> Tuple[float, float]:
        idx = np.searchsorted(fpr, target_far, side="right")
        if idx >= len(tpr):
            idx = len(tpr) - 1
        return float(tpr[idx]), float(thresholds[idx])

    tar_1pct, th_1pct = get_tar_at_far(0.01)
    tar_01pct, th_01pct = get_tar_at_far(0.001)

    # Detection and Identification Rate (DIR) at FAR = 1%
    if genuine_correct_id is not None and len(genuine_correct_id) == len(genuine_scores):
        gen_scores_arr = np.array(genuine_scores)
        gen_corr_arr = np.array(genuine_correct_id)
        # Probe is accepted (score >= th_1pct) AND identity is correct
        dir_1pct = float(np.mean((gen_scores_arr >= th_1pct) & gen_corr_arr))
    else:
        dir_1pct = tar_1pct

    # Sample ~50 smooth points for frontend charting (JSON friendly)
    sample_indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
    roc_points = [
        {
            "far": round(float(fpr[i]), 5),
            "tar": round(float(tpr[i]), 5),
            "threshold": 1.0 if np.isinf(thresholds[i]) else round(float(thresholds[i]), 4),
        }
        for i in sample_indices
    ]

    return {
        "auroc": round(roc_auc, 4),
        "tar_at_far_1pct": round(tar_1pct, 4),
        "tar_at_far_01pct": round(tar_01pct, 4),
        "dir_at_far_1pct": round(dir_1pct, 4),
        "threshold_at_far_1pct": 1.0 if np.isinf(th_1pct) else round(float(th_1pct), 4),
        "threshold_at_far_01pct": 1.0 if np.isinf(th_01pct) else round(float(th_01pct), 4),
        "roc_curve_points": roc_points,
    }
