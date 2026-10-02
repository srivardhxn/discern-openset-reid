"""
Open-set Decision Rule for Look-Alike Re-ID.
Implements the dual-barrier rejection criteria:
1. Adaptive similarity barrier: s1 >= tau_i
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
    Evaluates query similarity against enrolled prototypes using dual-barrier criteria.
    """
    def __init__(
        self,
        default_tau: float = 0.55,
        default_delta: float = 0.05,
        use_adaptive_tau: bool = True,
        use_margin_test: bool = True,
    ):
        self.default_tau = default_tau
        self.default_delta = default_delta
        self.use_adaptive_tau = use_adaptive_tau
        self.use_margin_test = use_margin_test

    def evaluate(
        self,
        query_embedding: np.ndarray,
        prototypes: List[IdentityPrototype],
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

        # Compute similarity to every enrolled identity prototype
        scores: List[Tuple[IdentityPrototype, float]] = []
        for p in prototypes:
            sim = p.similarity(query_embedding)
            scores.append((p, sim))

        # Sort descending by similarity
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
        elif self.use_adaptive_tau and hasattr(top_proto, "adaptive_tau"):
            tau = top_proto.adaptive_tau
        else:
            tau = self.default_tau

        delta = target_delta if target_delta is not None else (self.default_delta if self.use_margin_test else -1.0)

        # Dual-barrier checks
        passed_tau = (s1 >= tau)
        passed_margin = (margin >= delta)

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
