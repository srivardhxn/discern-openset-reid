#!/usr/bin/env python3
"""
package_for_submission.py
Creates a clean, production-ready compressed ZIP archive of the Discern project
for evaluation and email submission.

Excludes:
- .venv / virtual environments (~1.1 GB)
- frontend/node_modules (~111 MB)
- .git repository history
- models/discern_model.zip (duplicate archive ~283 MB)
- redundant PyTorch weights (.pt files ~200 MB, ONNX is used for deployment)
- __pycache__ and temporary cache files (.pytest_cache)
- scratch files
"""

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT.parent
ZIP_NAME = "discern_evaluation_submission.zip"
ZIP_PATH = OUTPUT_DIR / ZIP_NAME

EXCLUDE_DIRS = {
    ".venv",
    "node_modules",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "scratch",
    "models",  # redundant zip folder
    "dist",
    "build",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pt",  # redundant PyTorch checkpoints, ONNX embedder is primary
}

EXCLUDE_FILES = {
    "tatus --short",
    ".DS_Store",
}


def should_exclude(rel_path: Path) -> bool:
    parts = rel_path.parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    if rel_path.name in EXCLUDE_FILES:
        return True
    if rel_path.suffix in EXCLUDE_EXTENSIONS:
        return True
    return False


def build_zip(zip_path: Path, exclude_onnx: bool = False):
    print(f"\n[package] Creating: {zip_path.name} (exclude_onnx={exclude_onnx})")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            rel_dir = Path(root).relative_to(PROJECT_ROOT)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                rel_file = rel_dir / file
                if should_exclude(rel_file):
                    continue
                if exclude_onnx and rel_file.name == "discern_embedder.onnx":
                    continue
                
                abs_file = Path(root) / file
                archive_name = Path("discern") / rel_file
                zf.write(abs_file, str(archive_name))
                file_count += 1
                total_uncompressed_bytes += abs_file.stat().st_size

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    uncompressed_mb = total_uncompressed_bytes / (1024 * 1024)

    print(f"[SUCCESS] {zip_path.name}")
    print(f"   Files: {file_count} | Uncompressed: {uncompressed_mb:.2f} MB | ZIP Size: {zip_size_mb:.2f} MB")
    print(f"   Path: {zip_path.resolve()}")
    return zip_path


def build_submission_zips():
    full_zip = OUTPUT_DIR / "discern_evaluation_submission.zip"
    light_zip = OUTPUT_DIR / "discern_code_submission_under25mb.zip"

    # 1. Full package (with ONNX model weights)
    build_zip(full_zip, exclude_onnx=False)

    # 2. Lightweight package (for strict <25MB direct email attachment limit)
    build_zip(light_zip, exclude_onnx=True)


if __name__ == "__main__":
    build_submission_zips()
