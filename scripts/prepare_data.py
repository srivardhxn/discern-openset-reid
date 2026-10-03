"""
Data preparation script for Discern.
Sets up sample uniform dataset or reads existing Market-1501,
builds the open-set evaluation protocol split,
and auto-curates the LOW-VARIANCE subset via HSV clothing clustering.
"""

import os
import sys
import argparse
import json

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data.dataset import Market1501Dataset, SyntheticUniformDatasetGenerator
from backend.data.protocol import OpenSetProtocol
from backend.data.low_variance import curate_low_variance_subset

def main():
    parser = argparse.ArgumentParser(description="Prepare data and curate low-variance subset for Discern")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "sample_market1501"),
        help="Path to Market-1501 directory or synthetic dataset directory",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Force regeneration of synthetic uniform dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DISCERN - DATA PREPARATION & LOW-VARIANCE CURATION")
    print("=" * 60)

    dataset = Market1501Dataset(args.data_dir)
    if not dataset.exists() or args.force_synthetic:
        print(f"\n[1/3] Target dataset directory not found or forced at: {args.data_dir}")
        print("      Generating synthetic uniform look-alike dataset (36 identities in 3 uniform themes)...")
        gen = SyntheticUniformDatasetGenerator(
            output_dir=args.data_dir,
            num_identities=36,
            samples_per_id=10,
            seed=args.seed,
        )
        gen.generate(force=True)
        print(f"      [OK] Synthetic dataset successfully generated in: {args.data_dir}")
    else:
        print(f"\n[1/3] Using existing dataset at: {args.data_dir}")

    # Load splits
    train_samples = dataset.load_split("train")
    test_samples = dataset.load_split("test")
    query_samples = dataset.load_split("query")
    all_eval_samples = test_samples + query_samples

    print(f"\n      Loaded samples:")
    print(f"      - Training set : {len(train_samples)} images")
    print(f"      - Gallery/Test : {len(test_samples)} images")
    print(f"      - Query set    : {len(query_samples)} images")
    print(f"      - Total eval   : {len(all_eval_samples)} images")

    # Generate Open-Set Protocol Split
    print(f"\n[2/3] Constructing Seeded Open-Set Protocol Split (seed={args.seed})...")
    protocol = OpenSetProtocol(enrolled_ratio=0.5, gallery_samples_per_id=3, seed=args.seed)
    open_set_split = protocol.create_split(all_eval_samples)
    summary = open_set_split.summary()

    split_path = os.path.join(PROJECT_ROOT, "results", "open_set_split.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(open_set_split.to_dict(), f, indent=2)

    print("      Open-Set Split Summary:")
    print(f"      - Enrolled Identities   : {summary['num_enrolled_ids']}")
    print(f"      - Unenrolled Impostors  : {summary['num_unenrolled_ids']}")
    print(f"      - Gallery Samples       : {summary['num_gallery_samples']}")
    print(f"      - Validation Probes     : {summary['num_val_genuine_probes']} genuine, {summary['num_val_impostor_probes']} impostors")
    print(f"      - Test Probes           : {summary['num_test_genuine_probes']} genuine, {summary['num_test_impostor_probes']} impostors")
    print(f"      - Saved split to        : {split_path}")

    # Curate Low-Variance Look-Alike Subset
    print(f"\n[3/3] Curating Low-Variance Uniform Subset (HSV histogram clustering)...")
    lowvar_path = os.path.join(PROJECT_ROOT, "results", "lowvar_subset.json")
    lowvar_data = curate_low_variance_subset(
        samples=all_eval_samples,
        top_pairs_ratio=0.10,
        output_path=lowvar_path,
    )

    meta = lowvar_data["metadata"]
    print("      Low-Variance Subset Curation Complete:")
    print(f"      - Filtered Identities   : {meta['low_variance_identities_count']} of {meta['total_identities']}")
    print(f"      - Filtered Images       : {meta['low_variance_samples_count']} of {meta['total_samples']}")
    print(f"      - Appearance Clusters   : {meta['num_clusters']}")
    print(f"      - Top Look-alike Pairs  : {len(lowvar_data['top_lookalike_pairs'])}")
    print(f"      - Saved subset to       : {lowvar_path}")

    print("\n[OK] Phase 1 Data Preparation Finished Successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
