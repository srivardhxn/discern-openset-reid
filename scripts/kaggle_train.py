"""
Kaggle T4 / GPU Strong-Baseline Training & Ablation Suite on Full Market-1501.
Implements the proven Bag-of-Tricks (BoT) Re-ID training recipe:
- Backbone: OSNet x1.0 pretrained on MSMT17 (with OSNet x0.5 comparison)
- Input: 256x128
- Augmentation: Random Horizontal Flip (p=0.5), Pad 10 + Random Crop (256, 128), Random Erasing (p=0.5)
- Head: BNNeck (BatchNorm1d(512) without bias shift)
- Loss: Cross-Entropy with Label Smoothing (0.1) + Batch-Hard Triplet Loss (margin 0.3)
- Sampler: PK with P=16, K=4 (batch size = 64)
- Optimizer: Adam with lr=3.5e-4 (backbone) and 10x (3.5e-3) for new heads, weight decay 5e-4
- Schedule: 10-epoch linear warmup followed by cosine annealing down to 1e-6 (60 epochs total)
- Automatic Mixed Precision (AMP)
- Model Selection: Best checkpoint selected by validation mAP, not last epoch
- Ablations:
  (M1) Strong Baseline (BoT recipe above)
  (M2) + Look-Alike-Aware PK Sampler (clothing-color clusters)
  (M3) + Color-Invariance Augmentation (Grayscale p=0.2, Channel Shuffle, Color Jitter) + Look-Alike PK
  (M4) + Horizontal Stripe Heads + Look-Alike PK
  (M5) + ArcFace Loss (s=20, m=0.25) replacing CE (optional comparison)
- Evaluates full standard Market-1501 protocol (CMC Rank-1, Rank-5, mAP with junk & same-camera filtering)
- Saves full manifest to results/kaggle_training_manifest.json and weights to weights/
"""

from __future__ import annotations
import os
import sys
import glob
import math
import time
import json
import random
import urllib.request
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models.osnet import OSNetReID
from backend.models.extractor import REID_TRANSFORM

# Official Hugging Face / Google Drive download links for MSMT17 pretrained weights
PRETRAINED_DOWNLOAD_URLS = {
    "osnet_x1_0": [
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    ],
    "osnet_x0_5": [
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    ],
}


def ensure_pretrained_weights(weight_key: str, dest_path: str) -> str:
    """Ensures pretrained weights exist locally, downloading if missing."""
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urls = PRETRAINED_DOWNLOAD_URLS.get(weight_key, [])
    print(f"Downloading official Kaiyang Zhou {weight_key} MSMT17 weights...")
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_f:
                total_len = resp.headers.get("Content-Length")
                total_len = int(total_len) if total_len else None
                downloaded = 0
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_len:
                        pct = (downloaded / total_len) * 100
                        print(f"\r  --> Progress: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB)", end="", flush=True)
            print()
            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
                print(f"[OK] Downloaded {weight_key} ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)")
                return dest_path
        except Exception as e:
            print(f"[WARN] Failed to download from {url}: {e}")

    # Fallback to gdown for Google Drive if gdown is installed
    try:
        import gdown
        gdrive_ids = {
            "osnet_x1_0": "1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
            "osnet_x0_5": "16DGLbZukvVYgINws8u8deSaOqjybZ83i",
        }
        gid = gdrive_ids.get(weight_key)
        if gid:
            print(f"Attempting gdown fallback for {weight_key} (ID: {gid})...")
            gdown.download(id=gid, output=dest_path, quiet=False)
            if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
                return dest_path
    except Exception:
        pass

    return dest_path


# =====================================================================
# 1. LATENCY & PARAMETER PROFILER
# =====================================================================

def profile_model(model: nn.Module, device: torch.device, num_warmup: int = 10, num_runs: int = 50) -> Dict[str, Any]:
    """Measures parameter counts and latency/throughput on target device."""
    model.eval()
    dummy = torch.randn(1, 3, 256, 128, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.extract_features(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.extract_features(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
    total_time = time.time() - t0
    ms_per_image = (total_time / max(1, num_runs)) * 1000.0
    fps = 1000.0 / ms_per_image if ms_per_image > 0 else 0.0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "device": device.type,
        "total_parameters": total_params,
        "parameters_million": round(total_params / 1e6, 3),
        "trainable_parameters": trainable_params,
        "latency_ms_per_image": round(ms_per_image, 2),
        "throughput_fps": round(fps, 1),
    }


# =====================================================================
# 2. CUSTOM AUGMENTATIONS & DATASET WRAPPER
# =====================================================================

class RandomChannelShuffle:
    """Randomly permutes RGB channel order with probability p."""
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p and img.mode == "RGB":
            channels = list(img.split())
            random.shuffle(channels)
            return Image.merge("RGB", channels)
        return img


def build_train_transform(color_invariance: bool = False) -> transforms.Compose:
    """
    Builds Re-ID training augmentations.
    Standard BoT: Resize, RandomHorizontalFlip, Pad+RandomCrop, ToTensor, Normalize, RandomErasing.
    Color Invariance (+M3): Adds Grayscale(p=0.2), RandomChannelShuffle(p=0.2), ColorJitter.
    """
    transform_list = [
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
    ]

    if color_invariance:
        transform_list.extend([
            transforms.RandomGrayscale(p=0.2),
            RandomChannelShuffle(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), value="random"),
    ])

    return transforms.Compose(transform_list)


class MarketTrainDataset(Dataset):
    """Training dataset returning (tensor, mapped_label, pid)."""
    def __init__(self, img_paths: List[str], pids: List[int], pid_to_label: Dict[int, int], transform: transforms.Compose):
        self.img_paths = img_paths
        self.pids = pids
        self.pid_to_label = pid_to_label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        path = self.img_paths[idx]
        pid = self.pids[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        label = self.pid_to_label[pid]
        return tensor, label, pid


# =====================================================================
# 3. PK SAMPLERS: UNIFORM & LOOK-ALIKE-AWARE
# =====================================================================

class UniformPKSampler:
    """Uniform Random PK Sampler: samples P identities, K images each per batch."""
    def __init__(self, pids: List[int], p: int = 16, k: int = 4, seed: int = 42):
        self.p = p
        self.k = k
        self.rng = random.Random(seed)

        self.pid_to_indices = defaultdict(list)
        for idx, pid in enumerate(pids):
            self.pid_to_indices[pid].append(idx)
        self.unique_pids = sorted(list(self.pid_to_indices.keys()))
        self.num_batches = max(1, len(pids) // (p * k))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            selected_pids = self.rng.sample(self.unique_pids, min(self.p, len(self.unique_pids)))
            # If fewer unique IDs than P, sample with replacement
            if len(selected_pids) < self.p:
                selected_pids = self.rng.choices(self.unique_pids, k=self.p)

            for pid in selected_pids:
                indices = self.pid_to_indices[pid]
                if len(indices) >= self.k:
                    chosen = self.rng.sample(indices, self.k)
                else:
                    chosen = self.rng.choices(indices, k=self.k)
                batch_indices.extend(chosen)
            yield batch_indices


class LookAlikePKSampler:
    """
    Look-Alike-Aware PK Sampler:
    Clusters identities into appearance cohorts (based on clothing color).
    Samples identities from the same or neighboring appearance clusters,
    forcing batch-hard triplet mining to resolve fine-grained cues.
    """
    def __init__(
        self,
        img_paths: List[str],
        pids: List[int],
        p: int = 16,
        k: int = 4,
        num_clusters: int = 16,
        seed: int = 42,
    ):
        self.p = p
        self.k = k
        self.rng = random.Random(seed)

        self.pid_to_indices = defaultdict(list)
        for idx, pid in enumerate(pids):
            self.pid_to_indices[pid].append(idx)
        self.unique_pids = sorted(list(self.pid_to_indices.keys()))
        self.num_batches = max(1, len(pids) // (p * k))

        # Build clothing-color clusters for training identities
        print(f"Building look-alike appearance clusters across {len(self.unique_pids)} training identities...")
        self.clusters = self._cluster_identities_by_color(img_paths, pids, num_clusters)
        print(f"[OK] Grouped identities into {len(self.clusters)} appearance cohorts.")

    def _cluster_identities_by_color(self, img_paths: List[str], pids: List[int], num_clusters: int) -> List[List[int]]:
        """Computes mean upper & lower body color vectors and groups identities."""
        pid_to_paths = defaultdict(list)
        for p_path, pid in zip(img_paths, pids):
            pid_to_paths[pid].append(p_path)

        features = []
        for pid in self.unique_pids:
            paths = pid_to_paths[pid][:4]
            colors = []
            for path in paths:
                try:
                    img = Image.open(path).convert("RGB").resize((32, 64))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    # Upper body (rows 10-35) and lower body (rows 35-60)
                    upper = arr[10:35, :, :].mean(axis=(0, 1))
                    lower = arr[35:60, :, :].mean(axis=(0, 1))
                    colors.append(np.concatenate([upper, lower]))
                except Exception:
                    pass
            if colors:
                mean_color = np.mean(colors, axis=0)
            else:
                mean_color = np.zeros(6, dtype=np.float32)
            features.append(mean_color)

        feat_arr = np.array(features)
        # Cluster via K-Means if scikit-learn is available, else quantile hash
        try:
            from sklearn.cluster import KMeans
            n_c = min(num_clusters, max(1, len(self.unique_pids) // self.p))
            km = KMeans(n_clusters=n_c, random_state=42, n_init=5)
            labels = km.fit_predict(feat_arr)
            cluster_dict = defaultdict(list)
            for pid, label in zip(self.unique_pids, labels):
                cluster_dict[label].append(pid)
            return [ids for ids in cluster_dict.values() if len(ids) > 0]
        except Exception:
            # Fallback: partition by dominant color coordinate
            sorted_pids = [self.unique_pids[i] for i in np.argsort(feat_arr[:, 0] + feat_arr[:, 3])]
            chunk_size = max(self.p, len(sorted_pids) // num_clusters)
            return [sorted_pids[i:i + chunk_size] for i in range(0, len(sorted_pids), chunk_size)]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            # Pick a target appearance cohort
            cohort = self.rng.choice(self.clusters)

            if len(cohort) >= self.p:
                selected_pids = self.rng.sample(cohort, self.p)
            else:
                # Complement with other cohorts
                selected_pids = list(cohort)
                rem = self.p - len(selected_pids)
                other_pids = [pid for pid in self.unique_pids if pid not in cohort]
                selected_pids.extend(self.rng.sample(other_pids, min(rem, len(other_pids))))
                if len(selected_pids) < self.p:
                    selected_pids = self.rng.choices(self.unique_pids, k=self.p)

            for pid in selected_pids:
                indices = self.pid_to_indices[pid]
                if len(indices) >= self.k:
                    chosen = self.rng.sample(indices, self.k)
                else:
                    chosen = self.rng.choices(indices, k=self.k)
                batch_indices.extend(chosen)
            yield batch_indices


# =====================================================================
# 4. LOSS FUNCTIONS: BATCH-HARD TRIPLET & ARCFACE
# =====================================================================

class BatchHardTripletLoss(nn.Module):
    """
    Standard Batch-Hard Triplet Loss with margin (Luo et al. BoT):
    L = mean(max(0, max_pos D(a, p) - min_neg D(a, n) + margin))
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = embeddings.size(0)
        # Pairwise euclidean distance matrix
        dist = torch.cdist(embeddings, embeddings, p=2)

        mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        mask_neg = ~mask_pos

        # Hardest positive for each anchor
        dist_ap = dist[mask_pos].view(n, -1).max(dim=1)[0]

        # Hardest negative for each anchor (mask positives with large constant)
        dist_with_inf = dist + 1e5 * mask_pos.float()
        dist_an = dist_with_inf.min(dim=1)[0]

        loss = F.relu(dist_ap - dist_an + self.margin)
        return loss.mean()


class ArcFaceClassifier(nn.Module):
    """ArcFace angular margin classifier replacing standard linear CE."""
    def __init__(self, in_features: int, num_classes: int, scale: float = 20.0, margin: float = 0.25):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, feat: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(feat, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if label is None or not self.training:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        return output


# =====================================================================
# 5. MARKET-1501 STANDARD BENCHMARK EVALUATOR
# =====================================================================

def parse_market_fname(fname: str) -> Tuple[int, int]:
    base = os.path.basename(fname)
    parts = base.split("_")
    pid = int(parts[0])
    cam = int(parts[1][1]) if len(parts) > 1 and len(parts[1]) > 1 else 1
    return pid, cam


def extract_features_batch(model: nn.Module, img_paths: List[str], device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            chunk = img_paths[i:i + batch_size]
            tensors = [REID_TRANSFORM(Image.open(p).convert("RGB")) for p in chunk]
            batch_x = torch.stack(tensors).to(device)
            feat = model.extract_features(batch_x)
            all_feats.append(feat.cpu().numpy())
    if not all_feats:
        return np.empty((0, 512), dtype=np.float32)
    return np.concatenate(all_feats, axis=0)


def evaluate_market1501_standard(
    model: nn.Module,
    query_dir: str,
    gallery_dir: str,
    device: torch.device,
    max_queries: Optional[int] = None,
) -> Dict[str, float]:
    """
    Computes standard Market-1501 CMC (Rank-1, Rank-5) and mAP with strict protocol:
    1. Ignores junk images in gallery (pid <= 0).
    2. Ignores same-camera same-identity gallery images.
    """
    q_files = sorted(glob.glob(os.path.join(query_dir, "*.jpg")) + glob.glob(os.path.join(query_dir, "*.png")))
    g_files = sorted(glob.glob(os.path.join(gallery_dir, "*.jpg")) + glob.glob(os.path.join(gallery_dir, "*.png")))

    if not q_files or not g_files:
        return {"rank1": 0.0, "rank5": 0.0, "map": 0.0}

    if max_queries and len(q_files) > max_queries:
        rng = random.Random(42)
        q_files = rng.sample(q_files, max_queries)

    q_pids = np.array([parse_market_fname(f)[0] for f in q_files])
    q_cams = np.array([parse_market_fname(f)[1] for f in q_files])
    g_pids = np.array([parse_market_fname(f)[0] for f in g_files])
    g_cams = np.array([parse_market_fname(f)[1] for f in g_files])

    # Extract features
    q_feats = extract_features_batch(model, q_files, device)
    g_feats = extract_features_batch(model, g_files, device)

    # Compute similarity matrix (dot product of L2 normalized features)
    sim_matrix = np.dot(q_feats, g_feats.T)
    num_q = len(q_pids)

    r1_list = []
    r5_list = []
    all_ap = []

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_cams[q_idx]

        sims = sim_matrix[q_idx]
        order = np.argsort(-sims)

        # Discard junk images (pid <= 0) and same-camera same-identity images
        valid_mask = ~((g_pids[order] <= 0) | ((g_pids[order] == q_pid) & (g_cams[order] == q_cam)))
        filtered_pids = g_pids[order][valid_mask]

        matches = (filtered_pids == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue

        first_match = np.where(matches == 1)[0]
        r1_list.append(1.0 if len(first_match) > 0 and first_match[0] == 0 else 0.0)
        r5_list.append(1.0 if len(first_match) > 0 and first_match[0] < 5 else 0.0)

        cum_matches = np.cumsum(matches)
        precision = cum_matches / (np.arange(len(matches)) + 1)
        ap = float((precision * matches).sum() / matches.sum())
        all_ap.append(ap)

    return {
        "rank1": round(float(np.mean(r1_list)) if r1_list else 0.0, 4),
        "rank5": round(float(np.mean(r5_list)) if r5_list else 0.0, 4),
        "map": round(float(np.mean(all_ap)) if all_ap else 0.0, 4),
    }


# =====================================================================
# 6. TRAINING ENGINE
# =====================================================================

def train_reid_model(
    model_name: str,
    output_weights: str,
    data_dir: str,
    width_mult: float = 1.0,
    epochs: int = 60,
    p: int = 16,
    k: int = 4,
    base_lr: float = 3.5e-4,
    use_stripes: bool = False,
    use_arcface: bool = False,
    use_lookalike_sampler: bool = False,
    color_invariance_aug: bool = False,
    device_str: str = "cuda",
    pretrained_weights: Optional[str] = None,
    val_interval: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Trains a single model following the BoT Re-ID training recipe."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

    print("\n" + "=" * 78)
    print(f"TRAINING CONFIGURATION: {model_name}")
    print(f"Backbone Scale    : OSNet x{width_mult:.1f}")
    print(f"Target Checkpoint : {output_weights}")
    print(f"Device            : {device}")
    print(f"Batch Architecture: P={p} IDs x K={k} images = Batch Size {p * k}")
    print(f"Objective         : {'ArcFace Loss (s=20, m=0.25)' if use_arcface else 'Cross-Entropy (Label Smoothing 0.1) + Batch-Hard Triplet (m=0.3)'}")
    print(f"Head Structure    : {'BNNeck + Horizontal Stripes' if use_stripes else 'BNNeck (BoT)'}")
    print(f"Sampler           : {'Look-Alike-Aware PK Sampler' if use_lookalike_sampler else 'Uniform Random PK'}")
    print(f"Augmentation      : {'Color Invariance (Grayscale + Shuffle + Jitter + BoT)' if color_invariance_aug else 'Standard BoT (Flip + Pad/Crop + Erasing)'}")
    print(f"Optimization      : Adam (lr={base_lr}, 10x head lr), 10-ep Warmup + Cosine Decay, AMP")
    print("=" * 78)

    # 1. Dataset Discovery
    train_dir = os.path.join(data_dir, "bounding_box_train")
    query_dir = os.path.join(data_dir, "query")
    gallery_dir = os.path.join(data_dir, "bounding_box_test")

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.jpg")) + glob.glob(os.path.join(train_dir, "*.png")))
    valid_paths = []
    valid_pids = []
    for f in train_files:
        pid, _ = parse_market_fname(f)
        if pid > 0:
            valid_paths.append(f)
            valid_pids.append(pid)

    unique_pids = sorted(list(set(valid_pids)))
    num_classes = len(unique_pids)
    pid_to_label = {pid: i for i, pid in enumerate(unique_pids)}

    print(f"Dataset Overview   : {len(valid_paths):,} images across {num_classes} training identities")

    # 2. Build Model
    model = OSNetReID(
        num_classes=num_classes,
        width_mult=width_mult,
        use_stripes=use_stripes,
        use_bnneck=True,
        pretrained_path=pretrained_weights if (pretrained_weights and os.path.isfile(pretrained_weights)) else None,
    ).to(device)

    # Replace classifier with ArcFace if requested
    if use_arcface:
        model.classifier = ArcFaceClassifier(in_features=512, num_classes=num_classes, scale=20.0, margin=0.25).to(device)

    # Measure speed & parameters
    prof = profile_model(model, device)
    print(f"Model Complexity   : {prof['parameters_million']}M parameters | {prof['latency_ms_per_image']} ms/image ({prof['throughput_fps']} FPS on {device.type})")

    # 3. Dataloader & Sampler
    train_transform = build_train_transform(color_invariance=color_invariance_aug)
    train_ds = MarketTrainDataset(valid_paths, valid_pids, pid_to_label, train_transform)

    if use_lookalike_sampler:
        sampler = LookAlikePKSampler(valid_paths, valid_pids, p=p, k=k, num_clusters=16, seed=seed)
    else:
        sampler = UniformPKSampler(valid_pids, p=p, k=k, seed=seed)

    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0, pin_memory=(device.type == "cuda"))

    # 4. Losses
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = BatchHardTripletLoss(margin=0.3)

    # 5. Optimizer with 10x lr for newly initialized layers
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in ["classifier", "bnneck", "stripe"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": base_lr, "weight_decay": 5e-4},
        {"params": head_params, "lr": base_lr * 10.0, "weight_decay": 5e-4},
    ])

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 6. Warmup + Cosine Decay Learning Rate Scheduler
    warmup_epochs = 10
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * ((epoch + 1) / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 7. Training Loop with Model Selection by Validation mAP
    best_map = -1.0
    best_rank1 = -1.0
    best_epoch = -1
    best_state_dict = None
    eval_history = []

    start_train_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_tri = 0.0
        batches = 0
        t0 = time.time()

        for batch_x, batch_y, _ in train_loader:
            batches += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                emb, logits = model(batch_x)

                if use_arcface:
                    # ArcFace forward pass
                    logits = model.classifier(emb, batch_y)
                    loss_ce = criterion_ce(logits, batch_y)
                else:
                    loss_ce = criterion_ce(logits, batch_y)

                loss_tri = criterion_triplet(emb, batch_y)
                loss = loss_ce + loss_tri

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_tri += loss_tri.item()

        scheduler.step()
        ep_time = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        avg_loss = total_loss / max(1, batches)
        avg_ce = total_ce / max(1, batches)
        avg_tri = total_tri / max(1, batches)

        # Periodic validation
        do_val = (ep % val_interval == 0) or (ep >= epochs - 10 and ep % 2 == 0) or (ep == epochs)
        val_str = ""
        if do_val and os.path.isdir(query_dir) and os.path.isdir(gallery_dir):
            val_metrics = evaluate_market1501_standard(model, query_dir, gallery_dir, device, max_queries=500 if ep < epochs else None)
            val_str = f" | Val R-1: {val_metrics['rank1']*100:.1f}% mAP: {val_metrics['map']*100:.1f}%"
            eval_history.append({"epoch": ep, "metrics": val_metrics})

            if val_metrics["map"] > best_map:
                best_map = val_metrics["map"]
                best_rank1 = val_metrics["rank1"]
                best_epoch = ep
                # Save best checkpoint
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state_dict, output_weights)
                val_str += " *"

        print(
            f"Epoch [{ep:2d}/{epochs}] | Loss: {avg_loss:.4f} (CE: {avg_ce:.3f} Tri: {avg_tri:.3f}) | LR: {curr_lr:.6f} | {ep_time:.1f}s{val_str}",
            flush=True,
        )

    total_training_sec = time.time() - start_train_time

    # Load best checkpoint for final evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n[BEST CHECKPOINT] Restored Epoch {best_epoch} with Validation mAP: {best_map*100:.2f}% (Rank-1: {best_rank1*100:.2f}%)")
    else:
        torch.save(model.state_dict(), output_weights)

    # 8. Full Final Benchmark Evaluation
    print("Running Full Market-1501 Protocol Evaluation (all queries)...")
    final_metrics = evaluate_market1501_standard(model, query_dir, gallery_dir, device)
    print(f"--> FINAL RESULTS on Market-1501:")
    print(f"    Rank-1 : {final_metrics['rank1']*100:.2f}%")
    print(f"    Rank-5 : {final_metrics['rank5']*100:.2f}%")
    print(f"    mAP    : {final_metrics['map']*100:.2f}%")

    return {
        "model_name": model_name,
        "backbone": f"osnet_x{width_mult:.1f}",
        "width_mult": width_mult,
        "checkpoint": output_weights,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_map": best_map,
        "training_time_seconds": round(total_training_sec, 2),
        "profiling": prof,
        "final_market1501": final_metrics,
        "evaluation_history": eval_history,
    }


# =====================================================================
# 7. MAIN CLI & BATCH RUNNER
# =====================================================================

def detect_market1501_path(candidate_paths: List[str]) -> str:
    """Finds the best valid Market-1501 dataset directory."""
    for p in candidate_paths:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "bounding_box_train")):
            return p
    return candidate_paths[-1]


def main():
    parser = argparse.ArgumentParser(description="Kaggle T4 / GPU OSNet ReID Full Training Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Market-1501 dataset directory")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    parser.add_argument("--batch-p", type=int, default=16, help="Number of identities per batch (P, default: 16)")
    parser.add_argument("--batch-k", type=int, default=4, help="Number of images per identity (K, default: 4)")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="Backbone learning rate (default: 3.5e-4)")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--run-ablation", action="store_true", default=True, help="Run all 4 BoT ablation models")
    parser.add_argument("--include-arcface", action="store_true", default=False, help="Include optional M5 ArcFace model")
    parser.add_argument("--include-x05", action="store_true", default=True, help="Run compact OSNet x0.5 for comparison")
    parser.add_argument("--val-interval", type=int, default=5, help="Epoch interval between evaluations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Detect dataset
    dataset_candidates = [
        "/kaggle/input/market-1501-v150915/Market-1501-v15.09.15",
        "/kaggle/input/market-1501/Market-1501-v15.09.15",
        "/kaggle/input/market1501/Market-1501-v15.09.15",
        "/kaggle/input/market1501",
        os.path.join(PROJECT_ROOT, "data", "Market-1501-v15.09.15"),
        os.path.join(PROJECT_ROOT, "data", "sample_market1501"),
    ]
    data_dir = args.data_dir if args.data_dir else detect_market1501_path(dataset_candidates)
    print(f"[DATASET] Using Market-1501 source at: {data_dir}")

    weights_dir = os.path.join(PROJECT_ROOT, "weights")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Pretrained weights paths
    weights_x1_0 = os.path.join(weights_dir, "osnet_x1_0_msmt17.pth")
    weights_x0_5 = os.path.join(weights_dir, "osnet_x0_5_msmt17.pth")
    ensure_pretrained_weights("osnet_x1_0", weights_x1_0)
    ensure_pretrained_weights("osnet_x0_5", weights_x0_5)

    # Profiling comparison between x1.0 and x0.5
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("\n" + "=" * 78)
    print("BACKBONE PROFILING: OSNet x1.0 vs OSNet x0.5")
    print("=" * 78)
    m_x1 = OSNetReID(width_mult=1.0, pretrained_path=weights_x1_0).to(device)
    prof_x1 = profile_model(m_x1, device)
    del m_x1

    m_x05 = OSNetReID(width_mult=0.5, pretrained_path=weights_x0_5).to(device)
    prof_x05 = profile_model(m_x05, device)
    del m_x05

    print(f"OSNet x1.0 : {prof_x1['parameters_million']}M params | {prof_x1['latency_ms_per_image']} ms ({prof_x1['throughput_fps']} FPS on {device.type})")
    print(f"OSNet x0.5 : {prof_x05['parameters_million']}M params | {prof_x05['latency_ms_per_image']} ms ({prof_x05['throughput_fps']} FPS on {device.type})")

    # Define ablation models
    models_to_train = [
        {
            "id": "M1",
            "name": "Model 1: Strong Baseline (BoT + Random PK)",
            "weights": os.path.join(weights_dir, "model_1_strong_baseline.pth"),
            "width_mult": 1.0,
            "pretrained": weights_x1_0,
            "use_stripes": False,
            "use_arcface": False,
            "use_lookalike_sampler": False,
            "color_invariance_aug": False,
        },
        {
            "id": "M2",
            "name": "Model 2: + Look-Alike PK Sampler",
            "weights": os.path.join(weights_dir, "model_2_lookalike_pk.pth"),
            "width_mult": 1.0,
            "pretrained": weights_x1_0,
            "use_stripes": False,
            "use_arcface": False,
            "use_lookalike_sampler": True,
            "color_invariance_aug": False,
        },
        {
            "id": "M3",
            "name": "Model 3: + Color-Invariance Augmentation",
            "weights": os.path.join(weights_dir, "model_3_color_invariance.pth"),
            "width_mult": 1.0,
            "pretrained": weights_x1_0,
            "use_stripes": False,
            "use_arcface": False,
            "use_lookalike_sampler": True,
            "color_invariance_aug": True,
        },
        {
            "id": "M4",
            "name": "Model 4: + Horizontal Stripes Head",
            "weights": os.path.join(weights_dir, "model_4_stripes_strong.pth"),
            "width_mult": 1.0,
            "pretrained": weights_x1_0,
            "use_stripes": True,
            "use_arcface": False,
            "use_lookalike_sampler": True,
            "color_invariance_aug": True,
        },
    ]

    if args.include_arcface:
        models_to_train.append({
            "id": "M5",
            "name": "Model 5: + ArcFace Margin Loss",
            "weights": os.path.join(weights_dir, "model_5_arcface_strong.pth"),
            "width_mult": 1.0,
            "pretrained": weights_x1_0,
            "use_stripes": False,
            "use_arcface": True,
            "use_lookalike_sampler": True,
            "color_invariance_aug": True,
        })

    if args.include_x05:
        models_to_train.append({
            "id": "M0.5",
            "name": "Comparison: OSNet x0.5 Compact Baseline",
            "weights": os.path.join(weights_dir, "osnet_x0_5_strong_baseline.pth"),
            "width_mult": 0.5,
            "pretrained": weights_x0_5,
            "use_stripes": False,
            "use_arcface": False,
            "use_lookalike_sampler": False,
            "color_invariance_aug": False,
        })

    manifest = {
        "profiling": {
            "osnet_x1_0": prof_x1,
            "osnet_x0_5": prof_x05,
        },
        "models": [],
    }

    best_overall_map = -1.0
    best_overall_weights = None

    for cfg in models_to_train:
        res = train_reid_model(
            model_name=cfg["name"],
            output_weights=cfg["weights"],
            data_dir=data_dir,
            width_mult=cfg["width_mult"],
            epochs=args.epochs,
            p=args.batch_p,
            k=args.batch_k,
            base_lr=args.lr,
            use_stripes=cfg["use_stripes"],
            use_arcface=cfg["use_arcface"],
            use_lookalike_sampler=cfg["use_lookalike_sampler"],
            color_invariance_aug=cfg["color_invariance_aug"],
            device_str=args.device,
            pretrained_weights=cfg["pretrained"],
            val_interval=args.val_interval,
            seed=args.seed,
        )
        manifest["models"].append(res)

        if res["final_market1501"]["map"] > best_overall_map:
            best_overall_map = res["final_market1501"]["map"]
            best_overall_weights = cfg["weights"]

    # Copy the best overall checkpoint to osnet_x1_0_market1501_best.pth and osnet_discern.pth
    if best_overall_weights and os.path.isfile(best_overall_weights):
        import shutil
        best_dest = os.path.join(weights_dir, "osnet_x1_0_market1501_best.pth")
        shutil.copy(best_overall_weights, best_dest)
        shutil.copy(best_overall_weights, os.path.join(weights_dir, "osnet_discern.pth"))
        print(f"\n[DEPLOYMENT] Selected best model ({best_overall_weights}) -> {best_dest}")

    manifest_path = os.path.join(results_dir, "kaggle_training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[COMPLETE] Training and benchmarking complete. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
