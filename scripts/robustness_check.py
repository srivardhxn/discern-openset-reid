"""
Robustness and Perturbation Evaluation for Discern.
Tests 50 random probe images across 4 realistic surveillance corruptions:
1. Brightness degradation (+/- 35% illumination shift)
2. Gaussian motion/sensor blur (sigma=2.0)
3. Partial body occlusion (lower 40% occluded by turnstiles/desks)
4. Low-resolution downsampling (48x24 surveillance degradation)
Reports accuracy, false accept rate (FAR), and margin degradation under stress.
Saves to results/robustness.json.
"""

from __future__ import annotations
import os
import sys
import json
import random
from typing import Dict, List, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data.dataset import ReIDSample
from backend.models.extractor import FeatureExtractor
from backend.matching.matcher import DiscernMatcher


def apply_perturbation(img: Image.Image, p_type: str) -> Image.Image:
    if p_type == "clean":
        return img
    elif p_type == "brightness":
        # Simulate dim twilight / harsh fluorescent lighting
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.55)
    elif p_type == "blur":
        # Simulate camera motion blur or defocus
        return img.filter(ImageFilter.GaussianBlur(radius=2.5))
    elif p_type == "occlusion":
        # Simulate lower body occlusion (e.g. waist-high security turnstile or counter)
        arr = np.array(img).copy()
        h = arr.shape[0]
        arr[int(h * 0.60):, :, :] = 25  # Dark occluder
        return Image.fromarray(arr)
    elif p_type == "low_resolution":
        # Downsample to 48x24 CCTV resolution then upsample back
        small = img.resize((24, 48), Image.Resampling.BILINEAR)
        return small.resize(img.size, Image.Resampling.BILINEAR)
    return img


def run_robustness_benchmarks(
    split_path: str = os.path.join(PROJECT_ROOT, "results", "open_set_split.json"),
    num_eval_queries: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 65)
    print("DISCERN - ROBUSTNESS & STRESS-TEST BENCHMARKING (50 QUERIES)")
    print("=" * 65)

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    gallery_samples = [ReIDSample(**s) for s in split_data["gallery_samples"]]
    genuine_probes = [ReIDSample(**s) for s in split_data["genuine_probe_samples"]]
    impostor_probes = [ReIDSample(**s) for s in split_data["impostor_probe_samples"]]

    rng = random.Random(seed)
    # Select 25 genuine probes and 25 impostor probes for a total of 50
    sampled_gen = rng.sample(genuine_probes, min(25, len(genuine_probes)))
    sampled_imp = rng.sample(impostor_probes, min(25, len(impostor_probes)))

    weights_path = os.path.join(PROJECT_ROOT, "weights", "osnet_discern.pth")
    extractor = FeatureExtractor(weights_path=weights_path if os.path.isfile(weights_path) else None)
    matcher = DiscernMatcher(
        use_whitening=True,
        use_prototypes=True,
        use_adaptive_threshold=True,
        use_margin_test=True,
        use_calibration=True,
        default_tau=0.55,
        default_delta=0.05,
    )

    # Enroll gallery
    id_to_paths = {}
    for s in gallery_samples:
        if os.path.isfile(s.image_path):
            id_to_paths.setdefault(s.identity_id, []).append(s.image_path)

    for pid, paths in id_to_paths.items():
        embs = extractor.extract_batch(paths)
        matcher.enroll(pid, f"Identity {pid}", embs, paths)

    perturbations = [
        ("clean", "Clean Benchmark", "Unaltered high-resolution standard Re-ID crop"),
        ("brightness", "Low-Light / Shift", "35% illumination drop simulating dim hallway lighting"),
        ("blur", "Sensor Motion Blur", "Gaussian blur (radius 2.5) simulating rapid transit motion"),
        ("occlusion", "Partial Turnstile Occlusion", "Lower 40% blocked simulating turnstiles/desks"),
        ("low_resolution", "Low-Res CCTV", "Downsampled to 48x24 and scaled back to 256x128"),
    ]

    results = {}

    for p_key, p_name, p_desc in perturbations:
        gen_accepted = 0
        gen_correct_id = 0
        gen_margins = []
        gen_confidences = []

        for probe in sampled_gen:
            img = Image.open(probe.image_path).convert("RGB")
            p_img = apply_perturbation(img, p_key)
            emb = extractor.extract(p_img)
            res = matcher.match(emb)
            if res.decision == "ACCEPTED":
                gen_accepted += 1
                if res.predicted_id == probe.identity_id:
                    gen_correct_id += 1
            gen_margins.append(res.margin)
            gen_confidences.append(res.calibrated_confidence)

        imp_false_accepts = 0
        imp_confidences = []
        for probe in sampled_imp:
            img = Image.open(probe.image_path).convert("RGB")
            p_img = apply_perturbation(img, p_key)
            emb = extractor.extract(p_img)
            res = matcher.match(emb)
            if res.decision == "ACCEPTED":
                imp_false_accepts += 1
            imp_confidences.append(res.calibrated_confidence)

        tar = gen_accepted / len(sampled_gen)
        dir_1 = gen_correct_id / len(sampled_gen)
        far = imp_false_accepts / len(sampled_imp)
        mean_margin = float(np.mean(gen_margins))

        results[p_key] = {
            "name": p_name,
            "description": p_desc,
            "tar_rate": round(tar, 4),
            "dir_rank1": round(dir_1, 4),
            "false_accept_rate": round(far, 4),
            "average_margin": round(mean_margin, 4),
            "avg_confidence": round(float(np.mean(gen_confidences)), 4),
        }

        print(f"      {p_name:<28} | TAR: {tar*100:>5.1f}% | FAR: {far*100:>5.1f}% | Mean Margin: {mean_margin:.3f}")

    out_file = os.path.join(PROJECT_ROOT, "results", "robustness.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Saved robustness benchmarks to: {out_file}")
    print("=" * 65)
    return results


def main():
    run_robustness_benchmarks()


if __name__ == "__main__":
    main()
