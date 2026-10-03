import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from scripts.official_osnet import osnet_x0_5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ReID Transform (256x128, RGB, ImageNet mean/std)
REID_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(model, img_paths, batch_size=32):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            tensors = [REID_TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths]
            x = torch.stack(tensors).to(device)
            # Raw pretrained OSNet forward pass returns 512-d pooled features
            feat = model(x)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)

def parse_market_fname(fname):
    # e.g. 0001_c1s1_001051_00.jpg
    base = os.path.basename(fname)
    parts = base.split("_")
    pid = int(parts[0])
    cam = int(parts[1][1])
    return pid, cam

def eval_market1501(query_feats, query_pids, query_cams, gallery_feats, gallery_pids, gallery_cams):
    sim_matrix = np.dot(query_feats, gallery_feats.T)
    num_q = len(query_pids)
    
    r1_list = []
    r5_list = []
    all_AP = []
    
    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_cam = query_cams[q_idx]
        
        sims = sim_matrix[q_idx]
        order = np.argsort(-sims)
        
        # Remove same PID and same CamID (junk)
        valid_mask = ~((gallery_pids[order] == q_pid) & (gallery_cams[order] == q_cam))
        filtered_pids = gallery_pids[order][valid_mask]
        
        # Check matches
        matches = (filtered_pids == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue
            
        first_match = np.where(matches == 1)[0]
        r1_list.append(1.0 if len(first_match) > 0 and first_match[0] == 0 else 0.0)
        r5_list.append(1.0 if len(first_match) > 0 and first_match[0] < 5 else 0.0)
        
        # AP
        cum_matches = np.cumsum(matches)
        precision = cum_matches / (np.arange(len(matches)) + 1)
        ap = (precision * matches).sum() / matches.sum()
        all_AP.append(ap)
        
    rank1 = float(np.mean(r1_list)) if r1_list else 0.0
    rank5 = float(np.mean(r5_list)) if r5_list else 0.0
    mAP = float(np.mean(all_AP)) if all_AP else 0.0
    return rank1, rank5, mAP

def eval_open_set_auroc(model, split_path):
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        
    g_samples = split_data["gallery_samples"]
    gen_probes = split_data["test_genuine_probes"]
    imp_probes = split_data["test_impostor_probes"]
    
    g_feats = extract_features(model, [s["image_path"] for s in g_samples])
    gen_feats = extract_features(model, [s["image_path"] for s in gen_probes])
    imp_feats = extract_features(model, [s["image_path"] for s in imp_probes])
    
    # Compute max cosine similarity to gallery
    gen_sims = np.max(np.dot(gen_feats, g_feats.T), axis=1)
    imp_sims = np.max(np.dot(imp_feats, g_feats.T), axis=1)
    
    # AUROC via Wilcoxon-Mann-Whitney
    n_g = len(gen_sims)
    n_i = len(imp_sims)
    ranks = np.sum(gen_sims[:, None] > imp_sims[None, :]) + 0.5 * np.sum(gen_sims[:, None] == imp_sims[None, :])
    auroc = float(ranks / (n_g * n_i))
    
    # TAR at FAR=1%
    threshold = float(np.percentile(imp_sims, 99.0))
    tar_1pct = float(np.mean(gen_sims >= threshold))
    
    return auroc, tar_1pct

def main():
    print("=" * 60)
    print("SANITY CHECK: Evaluating RAW Pretrained OSNet x0.5 Backbone")
    print("=" * 60)
    
    # Load official model
    model = osnet_x0_5(num_classes=1000, pretrained=False)
    weights_path = os.path.join(PROJECT_ROOT, "weights", "osnet_x0_5_msmt17.pth")
    sd = torch.load(weights_path, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    m_sd = model.state_dict()
    m_sd.update(sd)
    model.load_state_dict(m_sd)
    model.to(device)
    model.eval()
    
    # 1. Evaluate standard Market-1501 query vs gallery
    q_dir = os.path.join(PROJECT_ROOT, "data", "sample_market1501", "query")
    t_dir = os.path.join(PROJECT_ROOT, "data", "sample_market1501", "bounding_box_test")
    
    q_files = [os.path.join(q_dir, f) for f in sorted(os.listdir(q_dir)) if f.endswith(".jpg")]
    t_files = [os.path.join(t_dir, f) for f in sorted(os.listdir(t_dir)) if f.endswith(".jpg")]
    
    print(f"Loaded {len(q_files)} query images, {len(t_files)} test/gallery images.")
    
    q_pids = np.array([parse_market_fname(f)[0] for f in q_files])
    q_cams = np.array([parse_market_fname(f)[1] for f in q_files])
    t_pids = np.array([parse_market_fname(f)[0] for f in t_files])
    t_cams = np.array([parse_market_fname(f)[1] for f in t_files])
    
    q_feats = extract_features(model, q_files)
    t_feats = extract_features(model, t_files)
    
    rank1, rank5, mAP = eval_market1501(q_feats, q_pids, q_cams, t_feats, t_pids, t_cams)
    print(f"Market-1501 Standard Protocol:")
    print(f"  Rank-1 Accuracy : {rank1 * 100:.2f}%")
    print(f"  Rank-5 Accuracy : {rank5 * 100:.2f}%")
    print(f"  mAP             : {mAP * 100:.2f}%")
    
    # 2. Open-set AUROC on our split
    split_path = os.path.join(PROJECT_ROOT, "results", "open_set_split.json")
    auroc, tar_1pct = eval_open_set_auroc(model, split_path)
    print(f"\nOpen-Set Protocol on Held-Out Split:")
    print(f"  Open-Set AUROC  : {auroc:.4f}")
    print(f"  TAR @ 1.0% FAR  : {tar_1pct * 100:.2f}%")
    
    res = {
        "model": "raw_osnet_x0_5_msmt17",
        "standard_market1501": {
            "rank1": rank1,
            "rank5": rank5,
            "mAP": mAP,
            "query_count": len(q_files),
            "gallery_count": len(t_files)
        },
        "open_set_benchmark": {
            "auroc": auroc,
            "tar_at_far_1pct": tar_1pct
        }
    }
    
    out_path = os.path.join(PROJECT_ROOT, "results", "sanity_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved sanity check results to {out_path}")

if __name__ == "__main__":
    main()
