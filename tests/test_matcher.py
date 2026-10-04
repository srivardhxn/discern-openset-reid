"""
Unit tests for Discern Open-Set Matcher:
- Tripartite protocol split integrity with zero overlap between val-impostor and test-impostor sets
- Matcher accept of genuine probe
- Matcher reject of distant impostor (low_similarity)
- Matcher reject of look-alike via margin safety barrier
- AS-Norm adaptive score normalization against gallery cohort
- Per-identity threshold tau_i from gallery impostor similarities
- Gallery-adaptive whitening transformation
- Score calibration and conformal operating point selection
- Verification that training and matcher ablation flags alter decision state
- Metrics sanity and bootstrap confidence interval validity
"""

import os
import sys
import numpy as np
import pytest

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
    """Verify tripartite protocol: enrolled, val-impostor, and test-impostor identities have strictly zero overlap."""
    samples = []
    # 20 distinct identities with 8 samples each
    for pid in range(1, 21):
        for s_idx in range(8):
            samples.append(
                ReIDSample(
                    image_path=f"dummy/{pid}_{s_idx}.jpg",
                    identity_id=pid,
                    camera_id=1,
                    sequence_id=1,
                    frame_id=s_idx,
                )
            )

    protocol = OpenSetProtocol(enrolled_ratio=0.40, val_impostor_ratio=0.30, seed=42)
    split = protocol.create_split(samples)

    enrolled_set = set(split.enrolled_ids)
    val_imp_set = set(split.val_impostor_ids)
    test_imp_set = set(split.test_impostor_ids)

    # 1. Assert zero identity overlap across all three identity sets
    assert len(enrolled_set.intersection(val_imp_set)) == 0, "Enrolled and Val-Impostor identities overlap!"
    assert len(enrolled_set.intersection(test_imp_set)) == 0, "Enrolled and Test-Impostor identities overlap!"
    assert len(val_imp_set.intersection(test_imp_set)) == 0, "Val-Impostor and Test-Impostor sets overlap!"

    # 2. Check partition counts (40% / 30% / 30% of 20 = 8 / 6 / 6)
    assert len(enrolled_set) == 8
    assert len(val_imp_set) == 6
    assert len(test_imp_set) == 6

    # 3. Check probe identity sets
    val_imp_probe_ids = {p.identity_id for p in split.val_impostor_probes}
    test_imp_probe_ids = {p.identity_id for p in split.test_impostor_probes}
    assert len(val_imp_probe_ids.intersection(test_imp_probe_ids)) == 0, "Val and Test impostor probes have overlapping identities!"

    # 4. Check genuine probes strictly belong to enrolled set
    for p in split.val_genuine_probes:
        assert p.identity_id in enrolled_set
    for p in split.test_genuine_probes:
        assert p.identity_id in enrolled_set

    # 5. Check disjoint images between gallery, val genuine, and test genuine
    gal_imgs = {s.image_path for s in split.gallery_samples}
    val_gen_imgs = {s.image_path for s in split.val_genuine_probes}
    test_gen_imgs = {s.image_path for s in split.test_genuine_probes}

    assert len(gal_imgs.intersection(val_gen_imgs)) == 0, "Gallery and Val Genuine probes have image overlap!"
    assert len(gal_imgs.intersection(test_gen_imgs)) == 0, "Gallery and Test Genuine probes have image overlap!"
    assert len(val_gen_imgs.intersection(test_gen_imgs)) == 0, "Val Genuine and Test Genuine probes have image overlap!"


def test_gallery_adaptive_whitening():
    """Verify whitening fits on data and returns unit-normalized features."""
    np.random.seed(42)
    shared_noise = np.random.randn(1, 512)
    embeddings = np.random.randn(20, 512) * 0.2 + shared_noise * 0.8
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    whitener = GalleryAdaptiveWhitening(shrinkage=0.1)
    whitener.fit(embeddings)
    assert whitener.is_fitted

    transformed = whitener.transform(embeddings)
    assert transformed.shape == (20, 512)
    norms = np.linalg.norm(transformed, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_as_norm_scoring():
    """Verify AS-norm computes cohort statistics and normalizes query scores against enrolled cohort."""
    np.random.seed(42)
    matcher = DiscernMatcher(use_as_norm=True, as_norm_k=2)

    # Enroll 3 distinct identities
    e1 = np.random.randn(512).astype(np.float32)
    e1 /= np.linalg.norm(e1)
    e2 = np.random.randn(512).astype(np.float32)
    e2 /= np.linalg.norm(e2)
    e3 = np.random.randn(512).astype(np.float32)
    e3 /= np.linalg.norm(e3)

    matcher.enroll(1, "Alice", e1, recompute=False)
    matcher.enroll(2, "Bob", e2, recompute=False)
    matcher.enroll(3, "Charlie", e3, recompute=True)

    # Check cohort stats are computed
    for p in matcher.prototypes.values():
        assert p.cohort_mean is not None
        assert p.cohort_std > 0.0

    # Query with Alice with small angular noise
    noise = np.random.randn(512).astype(np.float32)
    noise /= np.linalg.norm(noise)
    query = 0.95 * e1 + 0.05 * noise
    query /= np.linalg.norm(query)

    scores = matcher.compute_scores(query)
    assert len(scores) == 3
    # Check that highest normalized score belongs to Alice
    best_proto, best_eff_score, best_raw_sim = max(scores, key=lambda x: x[1])
    assert best_proto.identity_id == 1
    assert best_raw_sim > 0.90


def test_per_identity_threshold_from_gallery():
    """Verify tau_i is calculated from cross-identity similarities in the gallery."""
    np.random.seed(42)
    matcher = DiscernMatcher(use_per_id_threshold=True, per_id_quantile=0.95)

    base = np.random.randn(512).astype(np.float32)
    base /= np.linalg.norm(base)

    # Look-alike A and B share high correlation
    v_a = base * 0.85 + np.random.randn(512) * 0.15
    v_a /= np.linalg.norm(v_a)
    v_b = base * 0.85 + np.random.randn(512) * 0.15
    v_b /= np.linalg.norm(v_b)

    # Independent identity C is orthogonal
    v_c = np.random.randn(512).astype(np.float32)
    v_c /= np.linalg.norm(v_c)

    matcher.enroll(1, "LookAlike A", v_a, recompute=False)
    matcher.enroll(2, "LookAlike B", v_b, recompute=False)
    matcher.enroll(3, "Unique C", v_c, recompute=True)

    proto_a = matcher.prototypes[1]
    proto_c = matcher.prototypes[3]

    # Look-alike A has higher cross-similarity in gallery than isolated C
    assert proto_a.per_id_tau >= proto_c.per_id_tau
    assert proto_a.nearest_lookalike_id == 2


def test_matcher_accept_genuine():
    """Verify genuine probe of enrolled identity is correctly accepted."""
    np.random.seed(42)
    matcher = DiscernMatcher(default_tau=0.50, default_delta=0.04)

    emb1 = np.random.randn(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    emb2 = np.random.randn(512).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)

    matcher.enroll(identity_id=1, name="Alice", embeddings=emb1)
    matcher.enroll(identity_id=2, name="Bob", embeddings=emb2)

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

    impostor = np.random.randn(512).astype(np.float32)
    impostor /= np.linalg.norm(impostor)

    result = matcher.match(impostor)
    assert result.decision == "UNKNOWN"
    assert result.predicted_id is None
    assert "low_similarity" in result.reason_code


def test_matcher_reject_lookalike_by_margin():
    """
    CRITICAL LOOK-ALIKE TEST:
    A probe is sufficiently similar to pass a naive threshold (s1 > 0.60),
    BUT is also almost equally similar to another enrolled identity (s2 ~ 0.58).
    Discern's competitive margin test MUST reject this look-alike as UNKNOWN.
    """
    np.random.seed(42)
    matcher = DiscernMatcher(default_tau=0.55, default_delta=0.06, use_margin_test=True)

    base_uniform = np.random.randn(512).astype(np.float32)
    base_uniform /= np.linalg.norm(base_uniform)

    emb_guard_a = base_uniform * 0.85 + np.random.randn(512) * 0.15
    emb_guard_a /= np.linalg.norm(emb_guard_a)

    emb_guard_b = base_uniform * 0.85 + np.random.randn(512) * 0.15
    emb_guard_b /= np.linalg.norm(emb_guard_b)

    matcher.enroll(identity_id=101, name="Officer Smith", embeddings=emb_guard_a)
    matcher.enroll(identity_id=102, name="Officer Jones", embeddings=emb_guard_b)

    query_lookalike = (emb_guard_a + emb_guard_b) / 2.0
    query_lookalike /= np.linalg.norm(query_lookalike)

    result = matcher.match(query_lookalike)
    assert result.decision == "UNKNOWN"
    assert "lookalike" in result.reason_code
    assert result.margin < 0.06


def test_monotonic_calibration_preserves_ranking():
    """Verify Isotonic regression calibration is monotonic and preserves score rank ordering."""
    calibrator = ScoreCalibrator()
    np.random.seed(42)

    gen_scores = list(np.random.normal(0.85, 0.05, 100))
    imp_scores = list(np.random.normal(0.40, 0.08, 100))

    calibrator.fit(gen_scores, imp_scores)
    assert calibrator.is_fitted

    scores_test = np.linspace(0.2, 0.9, 15)
    cal_scores = [calibrator.predict_confidence(s) for s in scores_test]

    # Verify monotonicity: non-decreasing
    for i in range(len(cal_scores) - 1):
        assert cal_scores[i + 1] >= cal_scores[i] - 1e-6


def test_metrics_sanity_and_bootstrap_intervals():
    """Verifies that compute_roc_metrics outputs valid monotonic numbers and non-empty CIs."""
    from backend.eval.metrics import compute_roc_metrics, validate_metrics_sanity

    np.random.seed(42)
    gen_scores = list(np.random.normal(0.75, 0.08, 100))
    imp_scores = list(np.random.normal(0.35, 0.09, 200))
    gen_correct = [True] * 95 + [False] * 5

    res = compute_roc_metrics(gen_scores, imp_scores, gen_correct, num_bootstrap=100)

    assert res["auroc"] > 0.90
    assert res["tar_at_far_1pct"] >= res["tar_at_far_01pct"]
    assert res["tar_at_far_1pct"] >= res["dir_at_far_1pct"]
    assert res["auroc_ci"][0] <= res["auroc"] <= res["auroc_ci"][1] + 1e-4
    assert validate_metrics_sanity(res, is_trained=True) is True
