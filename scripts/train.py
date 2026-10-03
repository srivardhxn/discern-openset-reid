"""
Training script for Discern OSNet Re-ID backbone and ablation study checkpoints.

Supports fine-tuning Kaiyang Zhou official pretrained OSNet weights on Market-1501:
- Config 1: Baseline (Global pooling, CrossEntropy loss, Random PK sampler)
- Config 2: + Margin Loss (Global pooling, ArcFace loss, Random PK sampler)
- Config 3: + Look-Alike Mining (Global pooling, ArcFace loss, LookAlike PK sampler)
- Config 4: + Horizontal Stripes (Stripe pooling, ArcFace loss, LookAlike PK sampler)

Kaggle T4 / GPU Execution:
    python scripts/train.py --device cuda --epochs 10 --batch-p 8 --batch-k 4 --lr 0.0003 --train-all-ablations

Local CPU Fine-Tuning:
    python scripts/train.py --device cpu --epochs 2 --max-batches 25 --train-all-ablations
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import time
import shutil
from typing import Dict, Any, Optional

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


def train_single_model(
    data_dir: str,
    epochs: int = 2,
    p: int = 4,
    k: int = 4,
    lr: float = 0.0005,
    use_stripes: bool = True,
    use_margin_loss: bool = True,
    use_lookalike_sampler: bool = True,
    no_train: bool = False,
    freeze_backbone: bool = False,
    max_batches: Optional[int] = None,
    pretrained_weights: str = "weights/osnet_x0_5_msmt17.pth",
    output_weights: str = "weights/osnet_discern.pth",
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("\n" + "=" * 65)
    print(f"DISCERN - TRAINING MODEL: {os.path.basename(output_weights)}")
    print(f"Device        : {device}")
    print(f"Stripes Head  : {use_stripes}")
    print(f"Loss Objective: {'ArcFace Margin Loss' if use_margin_loss else 'Cross-Entropy'}")
    print(f"Batch Sampler : {'Look-Alike PK Mining' if use_lookalike_sampler else 'Uniform Random PK'}")
    print(f"Pretrained    : {pretrained_weights if os.path.isfile(pretrained_weights) else 'None (random)'}")
    print("=" * 65)

    dataset = Market1501Dataset(data_dir)
    train_samples = dataset.load_split("train")
    if not train_samples:
        raise RuntimeError(f"No training samples found in {data_dir}/bounding_box_train. Run scripts/prepare_data.py first!")

    # Unique identity mapping
    unique_pids = sorted(list(set(s.identity_id for s in train_samples)))
    pid_to_label = {pid: i for i, pid in enumerate(unique_pids)}
    num_classes = len(unique_pids)

    # Initialize model with pretrained weights
    pretrained_to_load = pretrained_weights if (pretrained_weights and os.path.isfile(pretrained_weights)) else None
    model = OSNetReID(
        num_classes=num_classes,
        use_stripes=use_stripes,
        pretrained_path=pretrained_to_load,
    ).to(device)

    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    if no_train:
        print("[REPORT] --no-train flag enabled: skipping SGD training passes.")
        print(f"[REPORT] Saving initialized weights to: {output_weights}")
        torch.save(model.state_dict(), output_weights)
        return {
            "model_path": output_weights,
            "trained": False,
            "use_stripes": use_stripes,
            "use_margin_loss": use_margin_loss,
            "use_lookalike_sampler": use_lookalike_sampler,
            "device": str(device),
        }

    # Optionally freeze earlier stages for ultra-fast local fine-tuning
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any(k in name for k in ["proj", "classifier", "conv5"]):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"      [Freezing Backbone] Optimizing {trainable:,} projection and head parameters.")

    # Set up DataLoader with Look-alike PK Sampler
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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=5e-4,
    )

    total_batches_per_ep = len(sampler)
    if max_batches is not None:
        total_batches_per_ep = min(total_batches_per_ep, max_batches)

    print(f"      Fine-tuning for {epochs} epochs ({total_batches_per_ep} batches/epoch, batch size {p*k})...")
    model.train()
    start_train_time = time.time()

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        batch_idx = 0
        t0 = time.time()
        for batch_x, batch_y, _ in loader:
            batch_idx += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            emb, logits = model(batch_x)

            loss_c = criterion_cls(logits, batch_y)
            loss_t = criterion_triplet(emb, batch_y)
            loss = loss_c + loss_t

            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

            if max_batches is not None and batch_idx >= max_batches:
                break

        dt = time.time() - t0
        avg_loss = ep_loss / max(1, batch_idx)
        print(f"      Epoch [{ep}/{epochs}] - Loss: {avg_loss:.4f} - Batches: {batch_idx} - Time: {dt:.2f}s")

    elapsed_total = time.time() - start_train_time
    torch.save(model.state_dict(), output_weights)
    print(f"[OK] Trained weights saved successfully to: {output_weights} (Took {elapsed_total:.1f}s)")

    return {
        "model_path": output_weights,
        "trained": True,
        "epochs": epochs,
        "batches_per_epoch": total_batches_per_ep,
        "final_loss": round(avg_loss, 4),
        "use_stripes": use_stripes,
        "use_margin_loss": use_margin_loss,
        "use_lookalike_sampler": use_lookalike_sampler,
        "device": str(device),
        "training_time_sec": round(elapsed_total, 2),
    }


def train_all_ablations(
    data_dir: str,
    epochs: int = 2,
    p: int = 4,
    k: int = 4,
    lr: float = 0.0005,
    no_train: bool = False,
    freeze_backbone: bool = False,
    max_batches: Optional[int] = 25,
    pretrained_weights: str = "weights/osnet_x0_5_msmt17.pth",
    device_name: Optional[str] = None,
    skip_existing: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 65)
    print("DISCERN - MULTI-MODEL ABLATION TRAINING PIPELINE")
    print("Training 4 distinct models across training-time configurations:")
    print("  1. Baseline          : Stripes=No , MarginLoss=No , LookAlike=No")
    print("  2. + Margin Loss     : Stripes=No , MarginLoss=Yes, LookAlike=No")
    print("  3. + Look-Alike      : Stripes=No , MarginLoss=Yes, LookAlike=Yes")
    print("  4. + Stripes (Full)  : Stripes=Yes, MarginLoss=Yes, LookAlike=Yes")
    print("=" * 65)

    configs = [
        {
            "name": "model_1_baseline",
            "use_stripes": False,
            "use_margin_loss": False,
            "use_lookalike_sampler": False,
            "out": os.path.join(PROJECT_ROOT, "weights", "model_1_baseline.pth"),
        },
        {
            "name": "model_2_margin_loss",
            "use_stripes": False,
            "use_margin_loss": True,
            "use_lookalike_sampler": False,
            "out": os.path.join(PROJECT_ROOT, "weights", "model_2_margin_loss.pth"),
        },
        {
            "name": "model_3_lookalike",
            "use_stripes": False,
            "use_margin_loss": True,
            "use_lookalike_sampler": True,
            "out": os.path.join(PROJECT_ROOT, "weights", "model_3_lookalike.pth"),
        },
        {
            "name": "model_4_stripes",
            "use_stripes": True,
            "use_margin_loss": True,
            "use_lookalike_sampler": True,
            "out": os.path.join(PROJECT_ROOT, "weights", "model_4_stripes.pth"),
        },
    ]

    manifest = {"models": [], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for cfg in configs:
        if skip_existing and os.path.isfile(cfg["out"]) and os.path.getsize(cfg["out"]) > 2_000_000:
            print(f"\n[INFO] Skipping {os.path.basename(cfg['out'])}, already exists on disk.")
            manifest["models"].append({
                "model_path": cfg["out"],
                "trained": True,
                "cached": True,
                "use_stripes": cfg["use_stripes"],
                "use_margin_loss": cfg["use_margin_loss"],
                "use_lookalike_sampler": cfg["use_lookalike_sampler"],
            })
            continue

        res = train_single_model(
            data_dir=data_dir,
            epochs=epochs,
            p=p,
            k=k,
            lr=lr,
            use_stripes=cfg["use_stripes"],
            use_margin_loss=cfg["use_margin_loss"],
            use_lookalike_sampler=cfg["use_lookalike_sampler"],
            no_train=no_train,
            freeze_backbone=freeze_backbone,
            max_batches=max_batches,
            pretrained_weights=pretrained_weights,
            output_weights=cfg["out"],
            device_name=device_name,
            seed=seed,
        )
        manifest["models"].append(res)

    # Copy model_4_stripes.pth as the main production weights osnet_discern.pth
    main_weights = os.path.join(PROJECT_ROOT, "weights", "osnet_discern.pth")
    shutil.copy(configs[3]["out"], main_weights)
    print(f"\n[OK] Copied Full Model ({configs[3]['out']}) -> {main_weights}")

    # Benchmark parameter count and latency for full model
    latency_output = os.path.join(PROJECT_ROOT, "results", "latency.json")
    bm_model = OSNetReID(use_stripes=True, pretrained_path=main_weights).eval()
    bench_cpu = benchmark_model(bm_model, device_name="cpu", warmup_iters=5, test_iters=25)
    bench_data = {"cpu": bench_cpu}
    if torch.cuda.is_available():
        bench_gpu = benchmark_model(bm_model, device_name="cuda", warmup_iters=10, test_iters=50)
        bench_data["gpu"] = bench_gpu
    with open(latency_output, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    manifest_path = os.path.join(PROJECT_ROOT, "results", "training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Training manifest saved to: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train Discern Re-ID Backbone & Ablations")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "sample_market1501"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-p", type=int, default=4)
    parser.add_argument("--batch-k", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--pretrained-weights", type=str, default=os.path.join(PROJECT_ROOT, "weights", "osnet_x0_5_msmt17.pth"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-stripes", action="store_true", default=True)
    parser.add_argument("--no-stripes", dest="use_stripes", action="store_false")
    parser.add_argument("--use-margin-loss", action="store_true", default=True)
    parser.add_argument("--no-margin-loss", dest="use_margin_loss", action="store_false")
    parser.add_argument("--use-lookalike-sampler", action="store_true", default=True)
    parser.add_argument("--no-lookalike-sampler", dest="use_lookalike_sampler", action="store_false")
    parser.add_argument("--no-train", action="store_true", help="Fallback mode: benchmark & export without SGD")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze lower conv layers for fast CPU fine-tuning")
    parser.add_argument("--unfreeze-all", dest="freeze_backbone", action="store_false", help="Full end-to-end unfreeze (recommended for Kaggle T4 / GPU)")
    parser.add_argument("--max-batches", type=int, default=25, help="Max batches per epoch for fast local fine-tuning")
    parser.add_argument("--all-batches", dest="max_batches", action="store_const", const=None, help="Train on all dataset batches")
    parser.add_argument("--train-all-ablations", action="store_true", help="Train all 4 distinct ablation models")
    parser.add_argument("--retrain-all", action="store_true", help="Force retrain all models even if they exist")
    parser.add_argument("--output-weights", type=str, default=os.path.join(PROJECT_ROOT, "weights", "osnet_discern.pth"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.train_all_ablations:
        train_all_ablations(
            data_dir=args.data_dir,
            epochs=args.epochs,
            p=args.batch_p,
            k=args.batch_k,
            lr=args.lr,
            no_train=args.no_train,
            freeze_backbone=args.freeze_backbone,
            max_batches=args.max_batches,
            pretrained_weights=args.pretrained_weights,
            device_name=args.device,
            skip_existing=not args.retrain_all,
            seed=args.seed,
        )
    else:
        train_single_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            p=args.batch_p,
            k=args.batch_k,
            lr=args.lr,
            use_stripes=args.use_stripes,
            use_margin_loss=args.use_margin_loss,
            use_lookalike_sampler=args.use_lookalike_sampler,
            no_train=args.no_train,
            freeze_backbone=args.freeze_backbone,
            max_batches=args.max_batches,
            pretrained_weights=args.pretrained_weights,
            output_weights=args.output_weights,
            device_name=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
