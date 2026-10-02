"""
Training script for Discern OSNet Re-ID backbone.
Supports modular ablation flags:
- --use-stripes
- --use-margin-loss
- --use-lookalike-sampler
- --no-train (benchmarks latency/parameters and creates weights without lengthy training)
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data.dataset import Market1501Dataset, ReIDSample
from backend.data.sampler import LookAlikePKSampler
from backend.models.osnet import OSNetReID
from backend.models.losses import ArcFaceLoss, BatchHardTripletLoss
from backend.models.extractor import REID_TRANSFORM, benchmark_model


class ReIDTorchDataset(Dataset):
    """PyTorch Dataset wrapper around ReIDSample list."""
    def __init__(self, samples: list[ReIDSample], pid_to_label: Dict[int, int]):
        self.samples = samples
        self.pid_to_label = pid_to_label
        self.transform = REID_TRANSFORM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        tensor = self.transform(img)
        label = self.pid_to_label[s.identity_id]
        return tensor, label, s.identity_id


def train_discern(
    data_dir: str,
    epochs: int = 5,
    p: int = 4,
    k: int = 4,
    lr: float = 0.0005,
    use_stripes: bool = True,
    use_margin_loss: bool = True,
    use_lookalike_sampler: bool = True,
    no_train: bool = False,
    output_weights: str = "weights/osnet_discern.pth",
    latency_output: str = "results/latency.json",
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DISCERN - MODEL TRAINING & BENCHMARKING")
    print(f"Device: {device}")
    print(f"Flags : use_stripes={use_stripes}, use_margin_loss={use_margin_loss}, use_lookalike_sampler={use_lookalike_sampler}")
    print("=" * 60)

    dataset = Market1501Dataset(data_dir)
    train_samples = dataset.load_split("train")
    if not train_samples:
        raise RuntimeError(f"No training samples found in {data_dir}/bounding_box_train. Run scripts/prepare_data.py first!")

    # Unique identity mapping
    unique_pids = sorted(list(set(s.identity_id for s in train_samples)))
    pid_to_label = {pid: i for i, pid in enumerate(unique_pids)}
    num_classes = len(unique_pids)

    print(f"Loaded {len(train_samples)} training samples across {num_classes} identities.")

    # Initialize model
    model = OSNetReID(num_classes=num_classes, use_stripes=use_stripes).to(device)

    # Benchmark parameter count and latency
    print("\n[1/3] Benchmarking Model Parameters and Inference Latency...")
    os.makedirs(os.path.dirname(latency_output), exist_ok=True)
    bench_cpu = benchmark_model(model, device_name="cpu", warmup_iters=5, test_iters=25)
    bench_results = {"cpu": bench_cpu}

    if torch.cuda.is_available():
        bench_gpu = benchmark_model(model, device_name="cuda", warmup_iters=10, test_iters=50)
        bench_results["gpu"] = bench_gpu
        print(f"      GPU Latency: {bench_gpu['latency_ms_per_image']} ms/image ({bench_gpu['throughput_fps']} FPS)")

    print(f"      CPU Latency: {bench_cpu['latency_ms_per_image']} ms/image ({bench_cpu['throughput_fps']} FPS)")
    print(f"      Parameters : {bench_cpu['parameters_million']}M ({bench_cpu['total_parameters']:,} weights)")

    with open(latency_output, "w", encoding="utf-8") as f:
        json.dump(bench_results, f, indent=2)
    print(f"      Saved latency benchmarks to: {latency_output}")

    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    if no_train:
        print("\n[2/3] --no-train flag enabled: skipping SGD training passes.")
        print(f"[3/3] Saving initialized backbone weights to: {output_weights}")
        torch.save(model.state_dict(), output_weights)
        print("[OK] Training & Export complete.")
        return bench_results

    # Set up DataLoader with Look-alike PK Sampler
    print("\n[2/3] Setting up Training Pipeline...")
    cluster_info_path = os.path.join(PROJECT_ROOT, "results", "lowvar_subset.json")
    sampler = LookAlikePKSampler(
        samples=train_samples,
        num_identities_per_batch=p,
        num_samples_per_identity=k,
        cluster_info_path=cluster_info_path if use_lookalike_sampler else None,
        use_lookalike=use_lookalike_sampler,
        seed=seed,
    )
    torch_ds = ReIDTorchDataset(train_samples, pid_to_label)
    loader = DataLoader(torch_ds, batch_sampler=sampler)

    # Set up Loss functions
    if use_margin_loss:
        criterion_cls = ArcFaceLoss(scale=24.0, margin=0.35)
    else:
        criterion_cls = nn.CrossEntropyLoss()

    criterion_triplet = BatchHardTripletLoss(margin=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    print(f"\n[3/3] Training for {epochs} epochs (Batch size={p*k}, Batches/epoch={len(sampler)})...")
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        batch_count = 0
        t0 = time.time()
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            emb, logits = model(batch_x)

            if use_margin_loss:
                loss_c = criterion_cls(logits, batch_y)
            else:
                loss_c = criterion_cls(logits, batch_y)

            loss_t = criterion_triplet(emb, batch_y)
            loss = loss_c + loss_t

            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            batch_count += 1

        dt = time.time() - t0
        avg_loss = ep_loss / max(1, batch_count)
        print(f"      Epoch [{ep}/{epochs}] - Loss: {avg_loss:.4f} - Time: {dt:.2f}s")

    torch.save(model.state_dict(), output_weights)
    print(f"\n[OK] Model successfully trained and saved to: {output_weights}")
    print("=" * 60)
    return bench_results


def main():
    parser = argparse.ArgumentParser(description="Train Discern Re-ID Backbone")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "sample_market1501"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-p", type=int, default=4)
    parser.add_argument("--batch-k", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--use-stripes", action="store_true", default=True)
    parser.add_argument("--no-stripes", dest="use_stripes", action="store_false")
    parser.add_argument("--use-margin-loss", action="store_true", default=True)
    parser.add_argument("--no-margin-loss", dest="use_margin_loss", action="store_false")
    parser.add_argument("--use-lookalike-sampler", action="store_true", default=True)
    parser.add_argument("--no-lookalike-sampler", dest="use_lookalike_sampler", action="store_false")
    parser.add_argument("--no-train", action="store_true", help="Fast test mode: skip training passes")
    parser.add_argument("--output-weights", type=str, default=os.path.join(PROJECT_ROOT, "weights", "osnet_discern.pth"))
    parser.add_argument("--latency-output", type=str, default=os.path.join(PROJECT_ROOT, "results", "latency.json"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_discern(
        data_dir=args.data_dir,
        epochs=args.epochs,
        p=args.batch_p,
        k=args.batch_k,
        lr=args.lr,
        use_stripes=args.use_stripes,
        use_margin_loss=args.use_margin_loss,
        use_lookalike_sampler=args.use_lookalike_sampler,
        no_train=args.no_train,
        output_weights=args.output_weights,
        latency_output=args.latency_output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
