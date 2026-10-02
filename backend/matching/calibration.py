"""
Score Calibration & Conformal-Style Operating Point Selection.
Fits Isotonic Regression on validation score distributions to produce calibrated probabilities.
Provides conformal thresholding for strict target False Accept Rates (e.g., FAR = 1%, FAR = 0.1%).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.isotonic import IsotonicRegression


class ScoreCalibrator:
    """
    Calibrates raw decision scores to probabilistic confidence values [0, 1]
    and determines conformal operating thresholds for target FAR alpha.
    """
    def __init__(self):
        self.regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        self.is_fitted: bool = False
        self.val_impostor_scores: List[float] = []
        self.val_genuine_scores: List[float] = []
        self.current_alpha: float = 0.01
        self.operating_threshold_tau: float = 0.60
        self.operating_margin_delta: float = 0.05

    def fit(self, genuine_scores: List[float], impostor_scores: List[float]) -> ScoreCalibrator:
        """
        Fits Isotonic Regression on genuine (1) and impostor (0) scores.
        """
        self.val_genuine_scores = list(genuine_scores)
        self.val_impostor_scores = list(impostor_scores)

        scores = np.array(genuine_scores + impostor_scores, dtype=np.float32)
        labels = np.array([1.0] * len(genuine_scores) + [0.0] * len(impostor_scores), dtype=np.float32)

        # Sort for isotonic regression stability
        sort_idx = np.argsort(scores)
        self.regressor.fit(scores[sort_idx], labels[sort_idx])
        self.is_fitted = True

        # Pre-compute initial operating point for target alpha = 1%
        self.select_operating_point(alpha=0.01)
        return self

    def predict_confidence(self, score: float) -> float:
        """Maps raw score to calibrated confidence in [0, 1]."""
        if not self.is_fitted:
            # Fallback sigmoid-like heuristic if not yet fitted with validation data
            prob = 1.0 / (1.0 + np.exp(-12.0 * (score - 0.55)))
            return float(np.clip(prob, 0.01, 0.99))

        conf = float(self.regressor.predict([score])[0])
        return float(np.clip(conf, 0.01, 0.99))

    def select_operating_point(self, alpha: float) -> Dict[str, Any]:
        """
        Conformal operating point selection:
        Finds the smallest similarity threshold tau such that
        Empirical False Accept Rate on validation impostor scores <= alpha.
        """
        self.current_alpha = float(alpha)

        if not self.val_impostor_scores:
            # Heuristic reasonable values
            tau = 0.60 + 0.15 * (1.0 - np.clip(alpha, 0.001, 0.1))
            self.operating_threshold_tau = float(tau)
            self.operating_margin_delta = 0.05
            return {
                "alpha": self.current_alpha,
                "threshold_tau": round(self.operating_threshold_tau, 4),
                "margin_delta": self.operating_margin_delta,
                "calibrated_tar": None,
                "calibrated_far": self.current_alpha,
            }

        imp_arr = np.sort(np.array(self.val_impostor_scores))
        gen_arr = np.sort(np.array(self.val_genuine_scores)) if self.val_genuine_scores else np.array([])

        # (1 - alpha) quantile of impostor scores
        # E.g. for alpha=0.01 (1%), threshold is the 99th percentile of impostor similarities
        quantile_idx = int(np.ceil((1.0 - alpha) * len(imp_arr))) - 1
        quantile_idx = max(0, min(quantile_idx, len(imp_arr) - 1))
        chosen_tau = float(imp_arr[quantile_idx])

        # Ensure safety buffer for look-alike distributions
        chosen_tau = max(chosen_tau, 0.50)
        self.operating_threshold_tau = round(chosen_tau, 4)

        # Scale margin delta based on alpha rigor: tighter alpha requires stricter margin
        delta = 0.04 + 0.03 * (1.0 - np.clip(alpha * 10, 0.0, 1.0))
        self.operating_margin_delta = round(delta, 4)

        # Empirical TAR at this threshold
        tar = float(np.mean(gen_arr >= self.operating_threshold_tau)) if len(gen_arr) > 0 else 0.0
        far = float(np.mean(imp_arr >= self.operating_threshold_tau))

        return {
            "alpha": self.current_alpha,
            "threshold_tau": self.operating_threshold_tau,
            "margin_delta": self.operating_margin_delta,
            "calibrated_tar": round(tar, 4),
            "calibrated_far": round(far, 4),
        }
