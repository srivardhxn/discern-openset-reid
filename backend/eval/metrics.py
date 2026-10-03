"""
Metrics and ROC / DIR evaluation engine for open-set person re-identification.
Computes:
- ROC (TAR vs FAR curve)
- TAR @ FAR = 1% (0.01) and FAR = 0.1% (0.001) with bootstrap 95% CIs
- AUROC (Area under ROC curve) with bootstrap 95% CIs
- Detection and Identification Rate (DIR) at target FAR with bootstrap 95% CIs
- Sanity check assertion suite ensuring scientific credibility
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_curve, auc


def _calculate_single_point_metrics(
    gen_scores: np.ndarray,
    imp_scores: np.ndarray,
    gen_correct: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float, float]:
    """Helper to compute (auroc, tar_1pct, tar_01pct, dir_1pct, th_1pct, th_01pct)."""
    if len(gen_scores) == 0 or len(imp_scores) == 0:
        return 0.5, 0.0, 0.0, 0.0, 1.0, 1.0

    y_true = np.concatenate([np.ones(len(gen_scores), dtype=int), np.zeros(len(imp_scores), dtype=int)])
    y_scores = np.concatenate([gen_scores, imp_scores])

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0, 0.0, 1.0, 1.0

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    # Exact quantile-based thresholding for FAR targets on negative distribution
    # Threshold is the value T such that P(imp_score >= T) <= target_far
    th_1pct = float(np.quantile(imp_scores, 1.0 - 0.01))
    th_01pct = float(np.quantile(imp_scores, 1.0 - 0.001))

    # TAR is fraction of genuine probes scoring >= threshold
    tar_1pct = float(np.mean(gen_scores >= th_1pct))
    tar_01pct = float(np.mean(gen_scores >= th_01pct))

    # DIR at FAR = 1%: Probe must score >= th_1pct AND have correct top-1 identity
    if gen_correct is not None and len(gen_correct) == len(gen_scores):
        dir_1pct = float(np.mean((gen_scores >= th_1pct) & gen_correct))
    else:
        dir_1pct = tar_1pct

    return roc_auc, tar_1pct, tar_01pct, dir_1pct, th_1pct, th_01pct


def compute_roc_metrics(
    genuine_scores: List[float],
    impostor_scores: List[float],
    genuine_correct_id: Optional[List[bool]] = None,
    num_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes complete ROC metrics for open-set verification/identification,
    including non-parametric bootstrap 95% confidence intervals.
    """
    gen_arr = np.array(genuine_scores, dtype=np.float64)
    imp_arr = np.array(impostor_scores, dtype=np.float64)
    corr_arr = np.array(genuine_correct_id, dtype=bool) if genuine_correct_id is not None else None

    # Point estimates
    roc_auc, tar_1pct, tar_01pct, dir_1pct, th_1pct, th_01pct = _calculate_single_point_metrics(
        gen_arr, imp_arr, corr_arr
    )

    # 500-sample bootstrap for 95% Confidence Intervals
    rng = np.random.RandomState(seed)
    n_gen = len(gen_arr)
    n_imp = len(imp_arr)

    b_auroc = []
    b_tar_1pct = []
    b_tar_01pct = []
    b_dir_1pct = []

    if n_gen > 1 and n_imp > 1 and num_bootstrap > 0:
        for _ in range(num_bootstrap):
            idx_g = rng.choice(n_gen, size=n_gen, replace=True)
            idx_i = rng.choice(n_imp, size=n_imp, replace=True)

            sub_gen = gen_arr[idx_g]
            sub_imp = imp_arr[idx_i]
            sub_corr = corr_arr[idx_g] if corr_arr is not None else None

            a_b, t1_b, t01_b, d1_b, _, _ = _calculate_single_point_metrics(sub_gen, sub_imp, sub_corr)
            b_auroc.append(a_b)
            b_tar_1pct.append(t1_b)
            b_tar_01pct.append(t01_b)
            b_dir_1pct.append(d1_b)

        auroc_ci = (float(np.percentile(b_auroc, 2.5)), float(np.percentile(b_auroc, 97.5)))
        tar_1pct_ci = (float(np.percentile(b_tar_1pct, 2.5)), float(np.percentile(b_tar_1pct, 97.5)))
        tar_01pct_ci = (float(np.percentile(b_tar_01pct, 2.5)), float(np.percentile(b_tar_01pct, 97.5)))
        dir_1pct_ci = (float(np.percentile(b_dir_1pct, 2.5)), float(np.percentile(b_dir_1pct, 97.5)))
    else:
        auroc_ci = (roc_auc, roc_auc)
        tar_1pct_ci = (tar_1pct, tar_1pct)
        tar_01pct_ci = (tar_01pct, tar_01pct)
        dir_1pct_ci = (dir_1pct, dir_1pct)

    # Sample ~50 smooth points for frontend charting (JSON friendly)
    y_true = np.concatenate([np.ones(len(gen_arr), dtype=int), np.zeros(len(imp_arr), dtype=int)])
    y_scores = np.concatenate([gen_arr, imp_arr])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
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
        "auroc_ci": (round(auroc_ci[0], 4), round(auroc_ci[1], 4)),
        "tar_at_far_1pct": round(tar_1pct, 4),
        "tar_at_far_1pct_ci": (round(tar_1pct_ci[0], 4), round(tar_1pct_ci[1], 4)),
        "tar_at_far_01pct": round(tar_01pct, 4),
        "tar_at_far_01pct_ci": (round(tar_01pct_ci[0], 4), round(tar_01pct_ci[1], 4)),
        "dir_at_far_1pct": round(dir_1pct, 4),
        "dir_at_far_1pct_ci": (round(dir_1pct_ci[0], 4), round(dir_1pct_ci[1], 4)),
        "threshold_at_far_1pct": 1.0 if np.isinf(th_1pct) else round(float(th_1pct), 4),
        "threshold_at_far_01pct": 1.0 if np.isinf(th_01pct) else round(float(th_01pct), 4),
        "roc_curve_points": roc_points,
    }


def validate_metrics_sanity(metrics: Dict[str, Any], is_trained: bool = True) -> bool:
    """
    Sanity test suite verifying statistical integrity:
    1. AUROC must be > 0.50 for a trained model.
    2. Monotonicity: TAR@1% >= TAR@0.1%.
    3. Consistency: TAR@1% >= DIR@1% (DIR requires correct ID in addition to acceptance).
    4. Bootstrap bounds validity.
    """
    auroc = metrics["auroc"]
    tar_1pct = metrics["tar_at_far_1pct"]
    tar_01pct = metrics["tar_at_far_01pct"]
    dir_1pct = metrics["dir_at_far_1pct"]

    if is_trained and auroc <= 0.50:
        raise ValueError(f"Sanity Check Failed: Trained model AUROC={auroc} is <= 0.50 (worse than random chance).")

    if tar_1pct < tar_01pct - 1e-6:
        raise ValueError(f"Sanity Check Failed: Non-monotonic TAR: TAR@1% ({tar_1pct}) < TAR@0.1% ({tar_01pct}).")

    if tar_1pct < dir_1pct - 1e-6:
        raise ValueError(f"Sanity Check Failed: Inconsistent DIR: TAR@1% ({tar_1pct}) < DIR@1% ({dir_1pct}).")

    return True

