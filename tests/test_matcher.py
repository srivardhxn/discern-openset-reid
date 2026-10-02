"""
Unit tests for Discern Open-Set Matcher:
- Open-set protocol split integrity
- Matcher accept of genuine probe
- Matcher reject of distant impostor (low_similarity)
- Matcher reject of look-alike via margin safety barrier
- Gallery-adaptive whitening transformation
- Score calibration and conformal operating point selection
"""

import os
import sys
import numpy as np
import pytest

# Ensure discern project root is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data.dataset import ReIDSample
from backend.data.protocol import OpenSetProtocol
from backend.matching.whitening import GalleryAdaptiveWhitening
from backend.matching.prototypes import IdentityPrototype
from backend.matching.decision import OpenSetDecisionEngine
from backend.matching.calibration import ScoreCalibrator
from backend.matching.matcher import DiscernMatcher


def test_open_set_split_integrity():
    """Verify enrolled and unenrolled identities have strictly zero overlap."""
    samples = []
    for pid in range(1, 11):
        for s_idx in range(4):
            samples.append(
                ReIDSample(
                    image_path=f"dummy/{pid}_{s_idx}.jpg",
                    identity_id=pid,
                    camera_id=1,
                    sequence_id=1,
                    frame_id=s_idx,
                )
            )

    protocol = OpenSetProtocol(enrolled_ratio=0.5, gallery_samples_per_id=2, seed=42)
    split = protocol.create_split(samples)

    enrolled_set = set(split.enrolled_ids)
    unenrolled_set = set(split.unenrolled_ids)

    # Disjoint check
    assert len(enrolled_set.intersection(unenrolled_set)) == 0
    assert len(enrolled_set) == 5
    assert len(unenrolled_set) == 5

    # Check probe identities
    for probe in split.genuine_probe_samples:
        assert probe.identity_id in enrolled_set

    for probe in split.impostor_probe_samples:
        assert probe.identity_id in unenrolled_set
        assert probe.identity_id not in enrolled_set


def test_gallery_adaptive_whitening():
    """Verify whitening fits on data and returns unit-normalized features."""
    np.random.seed(42)
    # Generate synthetic embeddings with high shared correlation in first 10 dimensions
    shared_noise = np.random.randn(1, 512)
    embeddings = np.random.randn(20, 512) * 0.2 + shared_noise * 0.8
    # L2 normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    whitener = GalleryAdaptiveWhitening(shrinkage=0.1)
    whitener.fit(embeddings)
    assert whitener.is_fitted

    transformed = whitener.transform(embeddings)
    assert transformed.shape == (20, 512)
    # Check unit norm
    norms = np.linalg.norm(transformed, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_matcher_accept_genuine():
    """Verify genuine probe of enrolled identity is correctly accepted."""
    np.random.seed(42)
    matcher = DiscernMatcher(default_tau=0.50, default_delta=0.04)

    # Identity 1 base embedding
    emb1 = np.random.randn(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    # Identity 2 base embedding (orthogonal)
    emb2 = np.random.randn(512).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)

    matcher.enroll(identity_id=1, name="Alice", embeddings=emb1)
    matcher.enroll(identity_id=2, name="Bob", embeddings=emb2)

    # Query is close to Alice with slight noise
    query = emb1 + np.random.randn(512) * 0.05
    query /= np.linalg.norm(query)

    result = matcher.match(query)
    assert result.decision == "ACCEPTED"
    assert result.predicted_id == 1
    assert result.predicted_name == "Alice"
    assert result.margin > 0.04


def test_matcher_reject_low_similarity():
    """Verify distant impostor probe is rejected with low_similarity."""
    np.random.seed(42)
    matcher = DiscernMatcher(default_tau=0.60, default_delta=0.04)

    emb1 = np.random.randn(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    matcher.enroll(identity_id=1, name="Alice", embeddings=emb1)

    # Impostor with random vector orthogonal to Alice
    impostor = np.random.randn(512).astype(np.float32)
    impostor /= np.linalg.norm(impostor)

    result = matcher.match(impostor)
    assert result.decision == "UNKNOWN"
    assert result.predicted_id is None
    assert "low_similarity" in result.reason_code


def test_matcher_reject_lookalike_by_margin():
    """
    CRITICAL LOOK-ALIKE TEST:
    A probe is sufficiently similar to pass a naive plain threshold (s1 > 0.60),
    BUT is also almost equally similar to another enrolled identity (s2 ~ 0.58).
    Discern's competitive margin test MUST reject this look-alike as UNKNOWN.
    """
    np.random.seed(42)
    # Plain matcher without margin test would accept
    # Discern matcher with margin test (delta = 0.06) MUST reject
    matcher = DiscernMatcher(default_tau=0.55, default_delta=0.06, use_margin_test=True)

    # Create two look-alike enrolled identities (e.g., security guards in same uniform)
    base_uniform = np.random.randn(512).astype(np.float32)
    base_uniform /= np.linalg.norm(base_uniform)

    emb_guard_a = base_uniform * 0.85 + np.random.randn(512) * 0.15
    emb_guard_a /= np.linalg.norm(emb_guard_a)

    emb_guard_b = base_uniform * 0.85 + np.random.randn(512) * 0.15
    emb_guard_b /= np.linalg.norm(emb_guard_b)

    matcher.enroll(identity_id=101, name="Officer Smith", embeddings=emb_guard_a)
    matcher.enroll(identity_id=102, name="Officer Jones", embeddings=emb_guard_b)

    # Look-alike query that lies right in between Guard A and Guard B
    query_lookalike = (emb_guard_a + emb_guard_b) / 2.0
    query_lookalike /= np.linalg.norm(query_lookalike)

    result = matcher.match(query_lookalike)

    # Verify look-alike rejection
    assert result.decision == "UNKNOWN"
    assert "lookalike" in result.reason_code
    assert result.margin < 0.06  # Margin barrier triggered


def test_score_calibration_and_conformal_alpha():
    """Verify Isotonic calibration and conformal FAR operating point selection."""
    calibrator = ScoreCalibrator()
    np.random.seed(42)

    genuine_scores = list(np.random.normal(0.85, 0.05, 100))
    impostor_scores = list(np.random.normal(0.40, 0.08, 100))

    calibrator.fit(genuine_scores, impostor_scores)
    assert calibrator.is_fitted

    # High genuine score -> high confidence
    conf_high = calibrator.predict_confidence(0.85)
    conf_low = calibrator.predict_confidence(0.35)
    assert conf_high > 0.80
    assert conf_low < 0.20

    # Test conformal operating point for alpha = 0.01 (1%)
    op_1pct = calibrator.select_operating_point(alpha=0.01)
    assert op_1pct["alpha"] == 0.01
    assert op_1pct["threshold_tau"] > 0.50
    assert op_1pct["calibrated_far"] <= 0.05
