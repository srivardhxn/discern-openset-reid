"""
Master Discern Open-Set Matcher.
Orchestrates Gallery-Adaptive Whitening, Multi-Exemplar Identity Prototypes,
Dual-Barrier Decision Logic, and Calibrated Confidence Scoring.
All core components have dedicated toggle flags for modular ablation testing.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from .whitening import GalleryAdaptiveWhitening
from .prototypes import IdentityPrototype
from .decision import OpenSetDecisionEngine, DecisionResult
from .calibration import ScoreCalibrator


@dataclass
class MatchResult:
    decision: str  # "ACCEPTED" | "UNKNOWN"
    predicted_id: Optional[int]
    predicted_name: Optional[str]
    calibrated_confidence: float
    raw_similarity: float
    competitor_similarity: float
    margin: float
    reason_code: str
    human_reason: str
    top_candidates: List[Dict[str, Any]]
    operating_point: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiscernMatcher:
    """
    Open-Set Re-Identification Matcher with Look-Alike False-Accept Prevention.
    """
    def __init__(
        self,
        use_whitening: bool = True,
        use_prototypes: bool = True,
        use_adaptive_threshold: bool = True,
        use_margin_test: bool = True,
        use_calibration: bool = True,
        default_tau: float = 0.55,
        default_delta: float = 0.05,
        shrinkage: float = 0.15,
        max_exemplars: int = 4,
    ):
        self.use_whitening = use_whitening
        self.use_prototypes = use_prototypes
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_margin_test = use_margin_test
        self.use_calibration = use_calibration

        self.default_tau = default_tau
        self.default_delta = default_delta
        self.max_exemplars = max_exemplars

        self.whitener = GalleryAdaptiveWhitening(shrinkage=shrinkage)
        self.decision_engine = OpenSetDecisionEngine(
            default_tau=default_tau,
            default_delta=default_delta,
            use_adaptive_tau=use_adaptive_threshold,
            use_margin_test=use_margin_test,
        )
        self.calibrator = ScoreCalibrator()

        # Enrolled raw data: identity_id -> dict
        self._gallery_data: Dict[int, Dict[str, Any]] = {}
        # Compiled prototypes: identity_id -> IdentityPrototype
        self.prototypes: Dict[int, IdentityPrototype] = {}

    def get_enrolled_count(self) -> int:
        return len(self.prototypes)

    def enroll(
        self,
        identity_id: int,
        name: str,
        embeddings: np.ndarray,
        image_paths: Optional[List[str]] = None,
        recompute: bool = True,
    ) -> Optional[IdentityPrototype]:
        """
        Enrolls an identity with one or more sample embeddings.
        Automatically triggers gallery re-fitting and prototype updates unless recompute=False.
        """
        embs = np.atleast_2d(embeddings).astype(np.float32)
        # Ensure input embeddings are L2 normalized
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-7
        embs = embs / norms

        paths = list(image_paths) if image_paths else []
        self._gallery_data[identity_id] = {
            "name": name,
            "embeddings": embs,
            "image_paths": paths,
        }

        if recompute:
            self.recompute_gallery()
            return self.prototypes[identity_id]
        return None

    def delete(self, identity_id: int) -> bool:
        """Deletes an enrolled identity from gallery and re-fits whitening & prototypes."""
        if identity_id in self._gallery_data:
            del self._gallery_data[identity_id]
            self.recompute_gallery()
            return True
        return False

    def clear(self) -> None:
        """Clears entire enrolled gallery."""
        self._gallery_data.clear()
        self.prototypes.clear()
        self.whitener.is_fitted = False

    def recompute_gallery(self) -> None:
        """
        Re-fits gallery-adaptive whitening and updates prototypes and per-identity adaptive thresholds.
        """
        self.prototypes.clear()
        if not self._gallery_data:
            self.whitener.is_fitted = False
            return

        # Collect all enrolled embeddings across all identities
        all_embs_list = []
        for d in self._gallery_data.values():
            all_embs_list.append(d["embeddings"])
        stacked_embs = np.concatenate(all_embs_list, axis=0)

        # 1. Fit Gallery-Adaptive Whitening
        if self.use_whitening and len(stacked_embs) >= 2:
            self.whitener.fit(stacked_embs)
        else:
            self.whitener.is_fitted = False

        # 2. Build Prototypes (applying whitening if active)
        for pid, data in self._gallery_data.items():
            raw_embs = data["embeddings"]
            if self.use_whitening and self.whitener.is_fitted:
                processed_embs = self.whitener.transform(raw_embs)
            else:
                processed_embs = raw_embs

            proto = IdentityPrototype(
                identity_id=pid,
                name=data["name"],
                embeddings=processed_embs,
                image_paths=data["image_paths"],
                max_exemplars=self.max_exemplars,
            )
            self.prototypes[pid] = proto

        # 3. Compute Cross-Identity Look-Alike Relationships and Per-Identity Adaptive Thresholds
        proto_list = list(self.prototypes.values())
        for i, proto_a in enumerate(proto_list):
            max_cross_sim = -1.0
            nearest_lookalike_id = None

            for j, proto_b in enumerate(proto_list):
                if i == j:
                    continue
                # Centroid cross-similarity
                sim = float(np.dot(proto_a.mean_embedding, proto_b.mean_embedding))
                if sim > max_cross_sim:
                    max_cross_sim = sim
                    nearest_lookalike_id = proto_b.identity_id

            proto_a.nearest_lookalike_id = nearest_lookalike_id
            proto_a.nearest_lookalike_sim = max(0.0, max_cross_sim)

            # Adaptive threshold tau_i: must be higher than the nearest enrolled competitor
            # with safety buffer (e.g., nearest_sim + 0.04) or default_tau, whichever is stricter
            if self.use_adaptive_threshold and nearest_lookalike_id is not None and max_cross_sim > 0.40:
                proto_a.adaptive_tau = float(np.clip(max_cross_sim + 0.04, self.default_tau, 0.90))
            else:
                proto_a.adaptive_tau = self.default_tau

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Executes open-set match pipeline on a single query embedding:
        1. Optional Gallery Whitening projection
        2. Similarity scoring against enrolled prototypes
        3. Dual-barrier decision rule (s1 >= tau_i AND margin >= delta)
        4. Calibrated confidence estimation
        """
        q = np.array(query_embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(q) + 1e-7
        q = q / norm

        # 1. Whitening transform
        if self.use_whitening and self.whitener.is_fitted:
            q_proc = self.whitener.transform(q)
        else:
            q_proc = q

        # 2. Decision rule
        prototypes_list = list(self.prototypes.values())
        target_tau = self.calibrator.operating_threshold_tau if (self.use_calibration and self.calibrator.is_fitted) else None
        target_delta = self.calibrator.operating_margin_delta if (self.use_calibration and self.calibrator.is_fitted) else None

        dec_res = self.decision_engine.evaluate(
            query_embedding=q_proc,
            prototypes=prototypes_list,
            target_tau=target_tau,
            target_delta=target_delta,
        )

        # 3. Calibrated Confidence
        if self.use_calibration:
            fused_score = dec_res.s1 + 0.5 * dec_res.margin
            confidence = self.calibrator.predict_confidence(fused_score)
        else:
            confidence = float(np.clip(dec_res.s1, 0.0, 1.0))

        op_point = {
            "alpha": self.calibrator.current_alpha,
            "threshold_tau": dec_res.threshold_tau,
            "margin_delta": dec_res.margin_delta,
        }

        return MatchResult(
            decision=dec_res.decision,
            predicted_id=dec_res.predicted_id,
            predicted_name=dec_res.predicted_name,
            calibrated_confidence=round(confidence, 4),
            raw_similarity=dec_res.s1,
            competitor_similarity=dec_res.s2,
            margin=dec_res.margin,
            reason_code=dec_res.reason_code,
            human_reason=dec_res.human_reason,
            top_candidates=dec_res.top_candidates,
            operating_point=op_point,
        )

    def set_operating_point(self, alpha: float) -> Dict[str, Any]:
        """Dynamically adjusts target false-accept rate alpha (e.g. 0.01, 0.001)."""
        op = self.calibrator.select_operating_point(alpha)
        self.decision_engine.default_tau = op["threshold_tau"]
        self.decision_engine.default_delta = op["margin_delta"]
        return op

    def get_gallery_summary(self) -> List[Dict[str, Any]]:
        """Returns summary of all enrolled identities for UI/API consumption."""
        return [proto.to_dict() for proto in self.prototypes.values()]
