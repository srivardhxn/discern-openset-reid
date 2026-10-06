"""
test_model_bundle.py — Verification of the Discern model bundle and decision rules.

Runs the bundled demo from model/demo and asserts:
- Known probes correctly accepted: 40/40 (100% TAR on enrolled)
- Look-alike stranger false accepts reported as measured (2/40 at balanced operating point)
- Correct calibrator, feature extraction, and decision config structure
"""

import json
import os
import sys
from pathlib import Path
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from discern_inference import Discern


def test_decision_config_structure():
    """Verify decision_config.json contains expected architecture and operating points."""
    config_path = MODEL_DIR / "decision_config.json"
    assert config_path.exists(), f"Missing {config_path}"
    
    with open(config_path, "r") as f:
        cfg = json.load(f)
    
    assert cfg["model"]["embedding_dim"] == 2048
    assert cfg["preprocess"]["input_size_hw"] == [288, 144]
    assert cfg["default_operating_point"] == "balanced"
    assert "strict" in cfg["calibrators"]["s1_margin_z"]["operating_points"]
    assert "balanced" in cfg["calibrators"]["s1_margin_z"]["operating_points"]
    assert "lenient" in cfg["calibrators"]["s1_margin_z"]["operating_points"]


def test_model_demo_replay():
    """
    Replay the bundled demo against model/discern_inference.py.
    Asserts known probes accepted 40/40 and records look-alike false accepts.
    """
    assert (MODEL_DIR / "discern_embedder.onnx").exists(), "ONNX model missing"
    model = Discern(str(MODEL_DIR))
    
    manifest_path = MODEL_DIR / "demo" / "demo_manifest.json"
    assert manifest_path.exists(), "demo_manifest.json missing"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Enroll gallery
    for g in manifest["gallery"]:
        img_path = MODEL_DIR / "demo" / g["file"]
        assert img_path.exists(), f"Gallery image missing: {img_path}"
        model.enroll(g["identity"], [Image.open(img_path)])
    
    assert len(model.rows) == 40, f"Expected 40 unique identities, got {len(model.rows)}"
    
    # Run probes
    ok = 0
    tot = 0
    fa = 0
    unk = 0
    
    for p in manifest["probes"]:
        img_path = MODEL_DIR / "demo" / p["file"]
        assert img_path.exists(), f"Probe image missing: {img_path}"
        r = model.identify(Image.open(img_path))
        
        if p["truth"] == "enrolled":
            tot += 1
            if r["identity"] == p["expected"]:
                ok += 1
        else:
            unk += 1
            if r["decision"] == "accept":
                fa += 1
    
    print(f"\n[test_model_bundle] known probes: {ok}/{tot} | look-alike strangers falsely accepted: {fa}/{unk}")
    
    # Assert exact match with demo specs
    assert tot == 40, f"Expected 40 enrolled probes, got {tot}"
    assert ok == 40, f"Expected 40/40 accepted enrolled probes, got {ok}/{tot}"
    assert unk == 40, f"Expected 40 look-alike probes, got {unk}"
    # Look-alike false accept count as printed by model demo (2 out of 40)
    assert fa == 2, f"Expected 2/40 look-alike false accepts, got {fa}/{unk}"
