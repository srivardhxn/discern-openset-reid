"""
Kaggle / Colab T4 GPU Full Training & Standard Re-ID Evaluation Pipeline.

Trains 4 distinct ablation checkpoints for 30+ epochs on Market-1501 with all layers unfrozen:
1. Baseline: OSNet x0.5 + CrossEntropy Loss + Uniform Random PK
2. + Margin Loss: OSNet x0.5 + ArcFace Margin Loss + Uniform Random PK
3. + Look-Alike Mining: OSNet x0.5 + ArcFace Margin Loss + Look-Alike PK Sampler
4. + Horizontal Stripes: OSNet x0.5 with Stripe Heads + ArcFace Loss + Look-Alike PK Sampler

Evaluates standard Market-1501 (Rank-1, Rank-5, mAP) and open-set benchmark metrics per model.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import time
from typing import Dict, Any, List, Tuple
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data.dataset import Market1501Dataset, ReIDSample
from backend.data.sampler import LookAlikePKSampler
from backend.models.osnet import OSNetReID
from backend.models.losses import ArcFaceLoss, BatchHardTripletLoss
from backend.models.extractor import REID_TRANSFORM


class ReIDDatasetWrapper(Dataset):
    def __init__(self, samples: List[ReIDSample], pid_to_label: Dict[int, int], is_train: bool = True):
        self.samples = samples
        self.pid_to_label = pid_to_label
        self.is_train = is_train

        # Standard Re-ID training augmentations (random horizontal flip + random erase/pad)
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                transforms.RandomCrop((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = REID_TRANSFORM

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        tensor = self.transform(img)
        label = self.pid_to_label[s.identity_id]
        return tensor, label, s.identity_id


def parse_market_fname(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname)
    parts = base.split("_")
    pid = int(parts[0])
    cam = int(parts[1][1])
    return pid, cam


def extract_features(model: OSNetReID, img_paths: List[str], device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i + batch_size]
            tensors = [REID_TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(tensors).to(device)
            feat = model.extract_features(x)
            all_feats.append(feat.cpu().numpy())
    if not all_feats:
        return np.empty((0, 512), dtype=np.float32)
    return np.concatenate(all_feats, axis=0)


def evaluate_market1501_standard(model: OSNetReID, data_dir: str, device: torch.device) -> Dict[str, float]:
    """Computes standard Market-1501 CMC (Rank-1, Rank-5) and mAP with junk filtering."""
    q_dir = os.path.join(data_dir, "query")
    t_dir = os.path.join(data_dir, "bounding_box_test")

    q_files = [os.path.join(q_dir, f) for f in sorted(os.listdir(q_dir)) if f.endswith(".jpg")]
    t_files = [os.path.join(t_dir, f) for f in sorted(os.listdir(t_dir)) if f.endswith(".jpg")]

    if not q_files or not t_files:
        return {"rank1": 0.0, "rank5": 0.0, "map": 0.0}

    q_pids = np.array([parse_market_fname(f)[0] for f in q_files])
    q_cams = np.array([parse_market_fname(f)[1] for f in q_files])
    t_pids = np.array([parse_market_fname(f)[0] for f in t_files])
    t_cams = np.array([parse_market_fname(f)[1] for f in t_files])

    q_feats = extract_features(model, q_files, device)
    t_feats = extract_features(model, t_files, device)

    sim_matrix = np.dot(q_feats, t_feats.T)
    num_q = len(q_pids)

    r1_list = []
    r5_list = []
    all_ap = []

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_cams[q_idx]

        sims = sim_matrix[q_idx]
        order = np.argsort(-sims)

        valid_mask = ~((t_pids[order] == q_pid) & (t_cams[order] == q_cam))
        filtered_pids = t_pids[order][valid_mask]

        matches = (filtered_pids == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue

        first_match = np.where(matches == 1)[0]
        r1_list.append(1.0 if len(first_match) > 0 and first_match[0] == 0 else 0.0)
        r5_list.append(1.0 if len(first_match) > 0 and first_match[0] < 5 else 0.0)

        cum_matches = np.cumsum(matches)
        precision = cum_matches / (np.arange(len(matches)) + 1)
        ap = (precision * matches).sum() / matches.sum()
        all_ap.append(ap)

    return {
        "rank1": float(np.mean(r1_list)) if r1_list else 0.0,
        "rank5": float(np.mean(r5_list)) if r5_list else 0.0,
        "map": float(np.mean(all_ap)) if all_ap else 0.0,
    }


def evaluate_open_set_auroc(model: OSNetReID, split_path: str, device: torch.device) -> Dict[str, float]:
    if not os.path.isfile(split_path):
        return {"auroc": 0.0, "tar_1pct": 0.0}

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    g_samples = split_data["gallery_samples"]
    gen_probes = split_data["test_genuine_probes"]
    imp_probes = split_data["test_impostor_probes"]

    g_feats = extract_features(model, [s["image_path"] for s in g_samples], device)
    gen_feats = extract_features(model, [s["image_path"] for s in gen_probes], device)
    imp_feats = extract_features(model, [s["image_path"] for s in imp_probes], device)

    gen_sims = np.max(np.dot(gen_feats, g_feats.T), axis=1)
    imp_sims = np.max(np.dot(imp_feats, g_feats.T), axis=1)

    n_g = len(gen_sims)
    n_i = len(imp_sims)
    ranks = np.sum(gen_sims[:, None] > imp_sims[None, :]) + 0.5 * np.sum(gen_sims[:, None] == imp_sims[None, :])
    auroc = float(ranks / (n_g * n_i)) if (n_g * n_i) > 0 else 0.5

    threshold = float(np.percentile(imp_sims, 99.0))
    tar_1pct = float(np.mean(gen_sims >= threshold))

    return {"auroc": auroc, "tar_1pct": tar_1pct}


def train_model(
    model_name: str,
    output_weights: str,
    data_dir: str,
    epochs: int = 30,
    p: int = 8,
    k: int = 4,
    lr: float = 0.0003,
    use_stripes: bool = False,
    use_margin_loss: bool = False,
    use_lookalike_sampler: bool = False,
    unfreeze_all: bool = True,
    device_str: str = "cuda",
    pretrained_weights: str = "weights/osnet_x0_5_msmt17.pth",
    seed: int = 42,
    max_batches: int = None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print(f"Target Checkpoint : {output_weights}")
    print(f"Device            : {device}")
    print(f"Epochs            : {epochs}")
    print(f"Batch Architecture: P={p} IDs x K={k} images = Batch Size {p*k}")
    print(f"Objective         : {'ArcFace Margin Loss' if use_margin_loss else 'Cross-Entropy Loss'}")
    print(f"Batch Sampler     : {'Look-Alike Mining PK' if use_lookalike_sampler else 'Uniform Random PK'}")
    print(f"Feature Head      : {'Horizontal Stripe Pooling' if use_stripes else 'Global Pooling'}")
    print("=" * 70)

    dataset = Market1501Dataset(data_dir)
    train_samples = dataset.load_split("train")
    if not train_samples:
        raise RuntimeError(f"No training samples found in {data_dir}/bounding_box_train!")

    unique_pids = sorted(list(set(s.identity_id for s in train_samples)))
    pid_to_label = {pid: i for i, pid in enumerate(unique_pids)}
    num_classes = len(unique_pids)

    model = OSNetReID(
        num_classes=num_classes,
        use_stripes=use_stripes,
        pretrained_path=pretrained_weights if os.path.isfile(pretrained_weights) else None,
    ).to(device)

    if not unfreeze_all:
        for name, param in model.named_parameters():
            if not any(k in name for k in ["fc", "stripe", "classifier"]):
                param.requires_grad = False

    trainable_params = [p_tens for p_tens in model.parameters() if p_tens.requires_grad]
    total_params = sum(p_tens.numel() for p_tens in model.parameters())
    active_params = sum(p_tens.numel() for p_tens in trainable_params)
    print(f"Model Parameters  : {total_params:,} total, {active_params:,} trainable")

    cluster_info_path = os.path.join(PROJECT_ROOT, "results", "lowvar_subset.json")
    sampler = LookAlikePKSampler(
        samples=train_samples,
        num_identities_per_batch=p,
        num_samples_per_identity=k,
        cluster_info_path=cluster_info_path if use_lookalike_sampler else None,
        use_lookalike=use_lookalike_sampler,
        seed=seed,
    )
    torch_ds = ReIDDatasetWrapper(train_samples, pid_to_label, is_train=True)
    loader = DataLoader(torch_ds, batch_sampler=sampler, num_workers=0)

    criterion_cls = ArcFaceLoss(scale=24.0, margin=0.35) if use_margin_loss else nn.CrossEntropyLoss()
    criterion_triplet = BatchHardTripletLoss(margin=0.3)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        batches = 0
        t0 = time.time()

        for batch_x, batch_y, _ in loader:
            batches += 1
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
            if max_batches is not None and batches >= max_batches:
                break

        scheduler.step()
        dt = time.time() - t0
        avg_loss = ep_loss / max(1, batches)
        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch [{ep:2d}/{epochs}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {dt:.1f}s", flush=True)

    total_train_sec = time.time() - start_time
    torch.save(model.state_dict(), output_weights)
    print(f"[OK] Saved model checkpoint to {output_weights} ({total_train_sec:.1f}s)")

    # Post-training evaluation
    print(f"  --> Evaluating on Standard Market-1501 Benchmark...")
    std_metrics = evaluate_market1501_standard(model, data_dir, device)
    print(f"      Rank-1: {std_metrics['rank1']*100:.2f}% | Rank-5: {std_metrics['rank5']*100:.2f}% | mAP: {std_metrics['map']*100:.2f}%")

    split_path = os.path.join(PROJECT_ROOT, "results", "open_set_split.json")
    print(f"  --> Evaluating on Open-Set Held-Out Protocol Split...")
    openset_metrics = evaluate_open_set_auroc(model, split_path, device)
    print(f"      Open-Set AUROC: {openset_metrics['auroc']:.4f} | TAR@1%FAR: {openset_metrics['tar_1pct']*100:.2f}%")

    return {
        "model_name": model_name,
        "checkpoint": output_weights,
        "epochs": epochs,
        "training_time_seconds": round(total_train_sec, 2),
        "standard_market1501": std_metrics,
        "open_set_benchmark": openset_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Kaggle T4 / GPU OSNet ReID Full Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/sample_market1501", help="Path to Market-1501 dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-p", type=int, default=8, help="Number of identities per batch (P)")
    parser.add_argument("--batch-k", type=int, default=4, help="Number of images per identity (K)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate (default: 0.0003)")
    parser.add_argument("--device", type=str, default="cuda", help="Target execution device (cuda or cpu)")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch (for quick testing)")
    parser.add_argument("--unfreeze-all", dest="unfreeze_all", action="store_true", default=None, help="Unfreeze entire backbone for end-to-end training")
    parser.add_argument("--freeze-backbone", dest="unfreeze_all", action="store_false", help="Freeze conv1-conv4 for fast training")
    parser.add_argument("--pretrained-weights", type=str, default="weights/osnet_x0_5_msmt17.pth", help="Official pretrained OSNet weights")
    args = parser.parse_args()

    # Default: unfreeze all on GPU, freeze on CPU unless explicitly requested
    if args.unfreeze_all is None:
        unfreeze_all = (args.device == "cuda")
    else:
        unfreeze_all = args.unfreeze_all

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    weights_dir = os.path.join(PROJECT_ROOT, "weights")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    pretrained_path = os.path.join(PROJECT_ROOT, args.pretrained_weights) if not os.path.isabs(args.pretrained_weights) else args.pretrained_weights

    # Define the 4 ablation configurations
    ablation_plans = [
        {
            "name": "Model 1: Baseline (CrossEntropy + Random PK)",
            "output_weights": os.path.join(weights_dir, "model_1_baseline.pth"),
            "use_stripes": False,
            "use_margin_loss": False,
            "use_lookalike_sampler": False,
        },
        {
            "name": "Model 2: + Margin Loss (ArcFace + Random PK)",
            "output_weights": os.path.join(weights_dir, "model_2_margin_loss.pth"),
            "use_stripes": False,
            "use_margin_loss": True,
            "use_lookalike_sampler": False,
        },
        {
            "name": "Model 3: + Look-Alike Mining (ArcFace + LookAlike PK)",
            "output_weights": os.path.join(weights_dir, "model_3_lookalike.pth"),
            "use_stripes": False,
            "use_margin_loss": True,
            "use_lookalike_sampler": True,
        },
        {
            "name": "Model 4: + Horizontal Stripes (ArcFace + LookAlike PK + Stripes)",
            "output_weights": os.path.join(weights_dir, "model_4_stripes.pth"),
            "use_stripes": True,
            "use_margin_loss": True,
            "use_lookalike_sampler": True,
        },
    ]

    manifest = []
    print("=" * 70)
    print("DISCERN - FULL KAGGLE T4 / GPU ABLATION TRAINING SUITE")
    print(f"Dataset: {data_dir}")
    print(f"Device : {args.device}")
    print(f"Epochs : {args.epochs} per model (Total {args.epochs * 4} epochs)")
    print("=" * 70)

    for plan in ablation_plans:
        res = train_model(
            model_name=plan["name"],
            output_weights=plan["output_weights"],
            data_dir=data_dir,
            epochs=args.epochs,
            p=args.batch_p,
            k=args.batch_k,
            lr=args.lr,
            use_stripes=plan["use_stripes"],
            use_margin_loss=plan["use_margin_loss"],
            use_lookalike_sampler=plan["use_lookalike_sampler"],
            unfreeze_all=unfreeze_all,
            device_str=args.device,
            pretrained_weights=pretrained_path,
            max_batches=args.max_batches,
        )
        manifest.append(res)

    # Copy Model 4 weights to osnet_discern.pth as the main discernment model
    import shutil
    shutil.copy(os.path.join(weights_dir, "model_4_stripes.pth"), os.path.join(weights_dir, "osnet_discern.pth"))

    manifest_path = os.path.join(results_dir, "kaggle_training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[COMPLETE] All 4 models trained and evaluated. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
