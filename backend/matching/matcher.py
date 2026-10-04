"""
Master Discern Open-Set Matcher.
Orchestrates:
1. Multi-Exemplar Identity Prototypes
2. Per-Identity Threshold tau_i from Gallery Impostor Distribution
3. AS-Norm (Adaptive Score Normalization)
4. Competitive Margin Test (s1 - s2 >= delta)
5. Gallery-Adaptive Whitening
6. Calibration & Conformal Threshold Selection

All components have dedicated toggle flags for modular ablation testing.
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
        use_whitening: bool = False,
        use_prototypes: bool = True,
        use_per_id_threshold: bool = False,
        use_as_norm: bool = False,
        use_margin_test: bool = False,
        use_calibration: bool = True,
        as_norm_k: int = 10,
        default_tau: float = 0.55,
        default_delta: float = 0.04,
        per_id_quantile: float = 0.95,
        per_id_offset: float = 0.0,
        shrinkage: float = 0.15,
        max_exemplars: int = 4,
    ):
        self.use_whitening = use_whitening
        self.use_prototypes = use_prototypes
        self.use_per_id_threshold = use_per_id_threshold
        self.use_as_norm = use_as_norm
        self.use_margin_test = use_margin_test
        self.use_calibration = use_calibration

        self.as_norm_k = as_norm_k
        self.default_tau = default_tau
        self.default_delta = default_delta
        self.per_id_quantile = per_id_quantile
        self.per_id_offset = per_id_offset
        self.max_exemplars = max_exemplars

        self.whitener = GalleryAdaptiveWhitening(shrinkage=shrinkage)
        self.decision_engine = OpenSetDecisionEngine(
            default_tau=default_tau,
            default_delta=default_delta,
            use_adaptive_tau=use_per_id_threshold,
            use_per_id_threshold=use_per_id_threshold,
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
        Re-fits gallery-adaptive whitening, builds prototypes, calculates
        cross-identity gallery impostor distributions for per-identity thresholds,
        and computes AS-norm cohort statistics.
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

        # 3. Cross-Identity Gallery Impostor Distribution (Per-Identity Threshold tau_i)
        proto_list = list(self.prototypes.values())
        for i, proto_a in enumerate(proto_list):
            max_cross_sim = -1.0
            nearest_lookalike_id = None
            cross_sims: List[float] = []

            for j, proto_b in enumerate(proto_list):
                if i == j:
                    continue
                # Compute cross-similarity between prototypes
                sim = float(np.dot(proto_a.mean_embedding, proto_b.mean_embedding))
                cross_sims.append(sim)
                if sim > max_cross_sim:
                    max_cross_sim = sim
                    nearest_lookalike_id = proto_b.identity_id

            proto_a.nearest_lookalike_id = nearest_lookalike_id
            proto_a.nearest_lookalike_sim = max(0.0, max_cross_sim)

            if cross_sims:
                arr_sims = np.array(cross_sims, dtype=np.float32)
                q_val = float(np.quantile(arr_sims, self.per_id_quantile))
                proto_a.gallery_impostor_quantile = q_val
                # tau_i set from quantile + offset (or clamped to reasonable bounds)
                proto_a.per_id_tau = float(np.clip(q_val + self.per_id_offset, self.default_tau, 0.95))
                proto_a.adaptive_tau = proto_a.per_id_tau
            else:
                proto_a.per_id_tau = self.default_tau
                proto_a.adaptive_tau = self.default_tau

            # 4. AS-Norm Enrolled Cohort Statistics
            if len(cross_sims) > 0:
                sorted_cohort = sorted(cross_sims, reverse=True)
                k = min(self.as_norm_k, len(sorted_cohort))
                proto_a.cohort_mean = float(np.mean(sorted_cohort[:k]))
                proto_a.cohort_std = float(np.std(sorted_cohort[:k]) + 1e-5)
            else:
                proto_a.cohort_mean = 0.0
                proto_a.cohort_std = 1.0

        # Update decision engine flags
        self.decision_engine.use_per_id_threshold = self.use_per_id_threshold
        self.decision_engine.use_adaptive_tau = self.use_per_id_threshold
        self.decision_engine.use_margin_test = self.use_margin_test

    def compute_scores(self, query_embedding: np.ndarray) -> List[Tuple[IdentityPrototype, float, float]]:
        """
        Computes scores for all enrolled identities against a query embedding.
        Returns list of (prototype, effective_score, raw_similarity).
        """
        q = np.array(query_embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(q) + 1e-7
        q = q / norm

        if self.use_whitening and self.whitener.is_fitted:
            q_proc = self.whitener.transform(q)
        else:
            q_proc = q

        prototypes = list(self.prototypes.values())
        if not prototypes:
            return []

        # 1. Compute raw prototype similarities
        raw_sims: List[Tuple[IdentityPrototype, float]] = []
        for p in prototypes:
            sim = p.similarity(q_proc, use_exemplars=self.use_prototypes)
            raw_sims.append((p, float(sim)))

        # 2. Optional AS-Norm (Adaptive Score Normalization)
        if self.use_as_norm and len(raw_sims) >= 2:
            sim_values = [s for _, s in raw_sims]
            sim_values.sort(reverse=True)
            k = min(self.as_norm_k, len(sim_values))
            mu_q = float(np.mean(sim_values[:k]))
            sigma_q = float(np.std(sim_values[:k]) + 1e-5)

            results = []
            for p, raw_s in raw_sims:
                norm_probe = (raw_s - mu_q) / sigma_q
                norm_enroll = (raw_s - p.cohort_mean) / p.cohort_std
                as_norm_score = 0.5 * (norm_probe + norm_enroll)
                results.append((p, float(as_norm_score), raw_s))
            return results
        else:
            return [(p, raw_s, raw_s) for p, raw_s in raw_sims]

    def match(
        self,
        query_embedding: np.ndarray,
        target_tau: Optional[float] = None,
        target_delta: Optional[float] = None,
    ) -> MatchResult:
        """
        Executes open-set match pipeline on a single query embedding:
        1. Optional Gallery Whitening projection
        2. Prototype similarity scoring (+ optional AS-norm)
        3. Decision rule (tau / tau_i and margin delta)
        4. Calibrated confidence estimation
        """
        scores = self.compute_scores(query_embedding)
        if not scores:
            return MatchResult(
                decision="UNKNOWN",
                predicted_id=None,
                predicted_name=None,
                calibrated_confidence=0.01,
                raw_similarity=0.0,
                competitor_similarity=0.0,
                margin=0.0,
                reason_code="gallery_empty",
                human_reason="Gallery is empty. Enroll identities first.",
                top_candidates=[],
                operating_point={
                    "alpha": self.calibrator.current_alpha,
                    "threshold_tau": self.default_tau,
                    "margin_delta": self.default_delta,
                },
            )

        candidate_scores = [(p, eff_s) for p, eff_s, _ in scores]
        raw_sim_map = {p.identity_id: raw_s for p, _, raw_s in scores}

        # Operating points
        eff_tau = target_tau
        if eff_tau is None and self.use_calibration and self.calibrator.is_fitted:
            eff_tau = self.calibrator.operating_threshold_tau

        eff_delta = target_delta
        if eff_delta is None and self.use_calibration and self.calibrator.is_fitted:
            eff_delta = self.calibrator.operating_margin_delta

        dec_res = self.decision_engine.evaluate(
            query_embedding=query_embedding,
            prototypes=list(self.prototypes.values()),
            candidate_scores=candidate_scores,
            target_tau=eff_tau,
            target_delta=eff_delta,
        )

        # Calibrated Confidence
        if self.use_calibration and self.calibrator.is_fitted:
            confidence = self.calibrator.predict_confidence(dec_res.s1)
        elif self.use_calibration:
            confidence = self.calibrator.predict_confidence(dec_res.s1)
        else:
            confidence = float(np.clip(dec_res.s1, 0.01, 0.99))

        op_point = {
            "alpha": self.calibrator.current_alpha,
            "threshold_tau": dec_res.threshold_tau,
            "margin_delta": dec_res.margin_delta,
        }

        # Candidate details with both raw and effective similarities
        top_candidates = []
        for cand in dec_res.top_candidates:
            cid = cand["identity_id"]
            cand_dict = dict(cand)
            cand_dict["effective_score"] = cand["similarity"]
            cand_dict["raw_similarity"] = raw_sim_map.get(cid, cand["similarity"])
            top_candidates.append(cand_dict)

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
            top_candidates=top_candidates,
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
