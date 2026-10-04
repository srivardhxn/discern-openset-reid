"""
Open-set Decision Rule for Look-Alike Person Re-ID.
Implements modular decision criteria:
1. Similarity barrier: s1 >= tau (or per-identity tau_i derived from gallery impostors)
2. Competitive margin barrier: s1 - s2 >= delta
Returns structured decision and machine-readable rejection codes.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from .prototypes import IdentityPrototype


@dataclass
class CandidateMatch:
    identity_id: int
    name: str
    similarity: float
    rank: int


@dataclass
class DecisionResult:
    decision: str  # "ACCEPTED" or "UNKNOWN"
    predicted_id: Optional[int]
    predicted_name: Optional[str]
    s1: float
    s2: float
    margin: float
    threshold_tau: float
    margin_delta: float
    reason_code: str
    human_reason: str
    top_candidates: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OpenSetDecisionEngine:
    """
    Evaluates candidate scores against enrolled prototypes using modular criteria:
    - Global or per-identity similarity threshold
    - Competitive margin barrier
    """
    def __init__(
        self,
        default_tau: float = 0.55,
        default_delta: float = 0.04,
        use_adaptive_tau: bool = False,
        use_per_id_threshold: bool = False,
        use_margin_test: bool = False,
    ):
        self.default_tau = default_tau
        self.default_delta = default_delta
        self.use_adaptive_tau = use_adaptive_tau or use_per_id_threshold
        self.use_per_id_threshold = use_per_id_threshold or use_adaptive_tau
        self.use_margin_test = use_margin_test

    def evaluate(
        self,
        query_embedding: np.ndarray,
        prototypes: List[IdentityPrototype],
        candidate_scores: Optional[List[Tuple[IdentityPrototype, float]]] = None,
        target_tau: Optional[float] = None,
        target_delta: Optional[float] = None,
    ) -> DecisionResult:
        if not prototypes:
            return DecisionResult(
                decision="UNKNOWN",
                predicted_id=None,
                predicted_name=None,
                s1=0.0,
                s2=0.0,
                margin=0.0,
                threshold_tau=self.default_tau,
                margin_delta=self.default_delta,
                reason_code="gallery_empty",
                human_reason="Gallery is empty. Enroll identities first.",
                top_candidates=[],
            )

        if candidate_scores is not None:
            scores = list(candidate_scores)
        else:
            scores = []
            for p in prototypes:
                sim = p.similarity(query_embedding)
                scores.append((p, sim))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        top_proto, s1 = scores[0]
        if len(scores) > 1:
            second_proto, s2 = scores[1]
            margin = s1 - s2
        else:
            second_proto = None
            s2 = -1.0
            margin = 1.0

        # Build top 3 candidate list
        top_candidates = []
        for rank, (proto, sim) in enumerate(scores[:3], start=1):
            top_candidates.append({
                "identity_id": proto.identity_id,
                "name": proto.name,
                "similarity": round(float(sim), 4),
                "rank": rank,
            })

        # Determine threshold tau
        if target_tau is not None:
            tau = target_tau
        elif self.use_per_id_threshold and hasattr(top_proto, "per_id_tau"):
            tau = top_proto.per_id_tau
        elif self.use_adaptive_tau and hasattr(top_proto, "adaptive_tau"):
            tau = top_proto.adaptive_tau
        else:
            tau = self.default_tau

        # Determine delta
        if target_delta is not None:
            delta = target_delta
        elif self.use_margin_test:
            delta = self.default_delta
        else:
            delta = -1.0  # Margin test disabled

        # Decision checks
        passed_tau = bool(s1 >= tau)
        passed_margin = bool(margin >= delta) if (self.use_margin_test or target_delta is not None) else True

        if passed_tau and passed_margin:
            decision = "ACCEPTED"
            reason_code = "accepted"
            human_reason = f"Accepted as {top_proto.name} (similarity {s1:.3f} >= {tau:.3f}, margin {margin:.3f} >= {delta:.3f})"
            predicted_id = top_proto.identity_id
            predicted_name = top_proto.name
        else:
            decision = "UNKNOWN"
            predicted_id = None
            predicted_name = None

            if not passed_tau and not passed_margin and second_proto is not None:
                reason_code = f"low_similarity_and_lookalike_of:{second_proto.identity_id}"
                human_reason = (
                    f"Rejected: similarity to {top_proto.name} ({s1:.3f} < {tau:.3f}) and "
                    f"too close to look-alike {second_proto.name} (margin {margin:.3f} < {delta:.3f})"
                )
            elif not passed_tau:
                reason_code = "low_similarity"
                human_reason = f"Rejected: similarity to {top_proto.name} ({s1:.3f}) is below threshold ({tau:.3f})"
            else:
                competitor_id = second_proto.identity_id if second_proto else "unknown"
                competitor_name = second_proto.name if second_proto else "another identity"
                reason_code = f"lookalike_of:{competitor_id}"
                human_reason = (
                    f"Rejected: look-alike ambiguity. Too close to {competitor_name} "
                    f"(margin {margin:.3f} is below safety threshold {delta:.3f})"
                )

        return DecisionResult(
            decision=decision,
            predicted_id=predicted_id,
            predicted_name=predicted_name,
            s1=round(float(s1), 4),
            s2=round(float(s2), 4),
            margin=round(float(margin), 4),
            threshold_tau=round(float(tau), 4),
            margin_delta=round(float(delta), 4),
            reason_code=reason_code,
            human_reason=human_reason,
            top_candidates=top_candidates,
        )
