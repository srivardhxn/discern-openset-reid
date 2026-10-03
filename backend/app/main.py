"""
FastAPI Backend for Discern Open-Set Re-Identification.
Provides endpoints for:
- POST /enroll (images + name)
- POST /match (image file or base64 -> decision, calibrated confidence, top-3 candidates, margin, reason)
- GET /gallery
- DELETE /identity/{id}
- PUT /operating-point (alpha)
- GET /metrics
- GET /roc
- GET /ablation
- GET /lookalikes
- Static file serving for crops and ROC charts
- Input validation, clear errors, and permissive CORS
"""

from __future__ import annotations
import os
import io
import sys
import json
import base64
import uuid
import shutil
import time
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models.extractor import FeatureExtractor
from backend.matching.matcher import DiscernMatcher
from backend.data.dataset import ReIDSample

app = FastAPI(
    title="Discern Open-Set Re-ID API",
    description="Minimizes false accepts under low inter-class appearance variance (uniform look-alikes).",
    version="1.0.0",
)

# Enable CORS for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "osnet_discern.pth")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount static files so UI can preview probe images, crops, and charts
app.mount("/static/data", StaticFiles(directory=DATA_DIR), name="static_data")
app.mount("/static/results", StaticFiles(directory=RESULTS_DIR), name="static_results")

# Global model and matcher instances
extractor: Optional[FeatureExtractor] = None
matcher: Optional[DiscernMatcher] = None


def init_engine():
    global extractor, matcher
    print("[INFO] Initializing Feature Extractor & Matcher Engine...")
    extractor = FeatureExtractor(weights_path=WEIGHTS_PATH if os.path.isfile(WEIGHTS_PATH) else None)
    matcher = DiscernMatcher(
        use_whitening=True,
        use_prototypes=True,
        use_adaptive_threshold=True,
        use_margin_test=True,
        use_calibration=True,
        default_tau=0.55,
        default_delta=0.05,
    )

    # Pre-populate gallery with benchmark enrolled identities if split exists
    split_path = os.path.join(RESULTS_DIR, "open_set_split.json")
    if os.path.isfile(split_path):
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                split_dict = json.load(f)
            gallery_samples = [ReIDSample(**s) for s in split_dict.get("gallery_samples", [])]

            valid_samples = [s for s in gallery_samples if os.path.isfile(s.image_path)]
            if valid_samples:
                all_paths = [s.image_path for s in valid_samples]
                all_embs = extractor.extract_batch(all_paths, batch_size=32)
                id_to_embs = {}
                id_to_paths = {}
                for idx, s in enumerate(valid_samples):
                    id_to_embs.setdefault(s.identity_id, []).append(all_embs[idx])
                    id_to_paths.setdefault(s.identity_id, []).append(s.image_path)

                for pid, embs in id_to_embs.items():
                    matcher.enroll(
                        identity_id=pid,
                        name=f"Identity {pid}",
                        embeddings=np.array(embs),
                        image_paths=id_to_paths[pid],
                        recompute=False,
                    )
                matcher.recompute_gallery()

            # Fit calibrator using genuine and impostor validation scores
            gen_samples = [s["image_path"] for s in split_dict.get("val_genuine_probes", split_dict.get("genuine_probe_samples", [])) if os.path.isfile(s["image_path"])]
            imp_samples = [s["image_path"] for s in split_dict.get("val_impostor_probes", split_dict.get("impostor_probe_samples", [])) if os.path.isfile(s["image_path"])]

            if gen_samples and imp_samples:
                g_embs = extractor.extract_batch(gen_samples[:32], batch_size=32)
                i_embs = extractor.extract_batch(imp_samples[:32], batch_size=32)
                g_scores = [matcher.match(e).raw_similarity for e in g_embs]
                i_scores = [matcher.match(e).raw_similarity for e in i_embs]
                matcher.calibrator.fit(g_scores, i_scores)

            print(f"[OK] Pre-enrolled {matcher.get_enrolled_count()} benchmark identities into gallery.")
        except Exception as e:
            print(f"[WARN] Failed to pre-populate gallery: {e}")


@app.on_event("startup")
def startup_event():
    init_engine()


# Pydantic Schemas
class OperatingPointUpdate(BaseModel):
    alpha: float = Field(..., ge=0.0001, le=0.5, description="Target False Accept Rate, e.g. 0.01 for 1%")


class MatchBase64Request(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string (JPEG/PNG)")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Discern Open-Set Re-Identification API",
        "version": "1.0.0",
        "enrolled_identities": matcher.get_enrolled_count() if matcher else 0,
        "operating_point": {
            "alpha": matcher.calibrator.current_alpha if matcher else 0.01,
            "threshold_tau": matcher.calibrator.operating_threshold_tau if matcher else 0.60,
            "margin_delta": matcher.calibrator.operating_margin_delta if matcher else 0.05,
        } if matcher else {},
    }


@app.get("/gallery")
def get_gallery():
    """Returns all enrolled identities in the gallery."""
    if not matcher:
        return []
    gallery = matcher.get_gallery_summary()
    # Normalize paths to URL accessible endpoints
    for item in gallery:
        norm_urls = []
        for p in item.get("image_paths", []):
            rel = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
            norm_urls.append(f"/static/{rel}")
        item["image_urls"] = norm_urls
    return gallery


@app.post("/enroll")
async def enroll_identity(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Enrolls a new identity into the open-set gallery."""
    global matcher, extractor
    if not matcher or not extractor:
        raise HTTPException(status_code=500, detail="Matching engine is not initialized")

    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Identity name cannot be empty")

    # Check for duplicate name
    existing_names = [p.name.strip().lower() for p in matcher.prototypes.values()]
    if clean_name.lower() in existing_names:
        raise HTTPException(status_code=400, detail=f"Identity '{clean_name}' is already enrolled in the gallery.")

    if not files:
        raise HTTPException(status_code=400, detail="At least one image must be provided")

    MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

    # Validate each file before creating folder or saving
    validated_images = []
    for file in files:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is empty (0 bytes).")
        if len(contents) > MAX_FILE_BYTES:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' exceeds maximum allowed size (10MB).")

        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a valid or readable image.")

        # Check visual variance (reject completely blank / solid color / non-person crops)
        img_arr = np.array(img)
        if float(np.std(img_arr)) < 3.0:
            raise HTTPException(
                status_code=400,
                detail=f"Image '{file.filename}' is blank or contains no subject (insufficient visual variance)."
            )

        # Scale down very large dimension images smoothly
        if img.width > 2048 or img.height > 2048:
            img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

        validated_images.append((file.filename, img))

    # Generate a unique identity ID (positive integer)
    existing_ids = list(matcher.prototypes.keys())
    new_id = (max(existing_ids) + 1) if existing_ids else 1

    id_folder = os.path.join(UPLOADS_DIR, f"id_{new_id}_{uuid.uuid4().hex[:6]}")
    os.makedirs(id_folder, exist_ok=True)

    saved_paths = []
    try:
        for idx, (orig_filename, img) in enumerate(validated_images):
            save_path = os.path.join(id_folder, f"{uuid.uuid4().hex[:8]}.jpg")
            img.save(save_path, "JPEG", quality=95)
            saved_paths.append(save_path)

        # Extract embeddings and enroll
        embeddings = extractor.extract_batch(saved_paths)
        proto = matcher.enroll(
            identity_id=new_id,
            name=clean_name,
            embeddings=embeddings,
            image_paths=saved_paths,
        )
    except Exception as e:
        shutil.rmtree(id_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    proto_dict = proto.to_dict()
    proto_dict["image_urls"] = [
        f"/static/{os.path.relpath(p, PROJECT_ROOT).replace('\\', '/')}" for p in saved_paths
    ]
    return {
        "status": "success",
        "message": f"Successfully enrolled '{clean_name}' with {len(saved_paths)} images.",
        "identity": proto_dict,
    }


@app.delete("/identity/{identity_id}")
@app.delete("/gallery/{identity_id}")
def delete_identity(identity_id: int):
    """Removes an identity from the enrolled gallery and re-fits whitening & prototypes."""
    global matcher
    if not matcher:
        raise HTTPException(status_code=500, detail="Matching engine not initialized")

    if identity_id not in matcher._gallery_data:
        raise HTTPException(status_code=404, detail=f"Identity {identity_id} not found in gallery")

    success = matcher.delete(identity_id)
    return {
        "status": "success" if success else "failed",
        "deleted_identity_id": identity_id,
        "remaining_gallery_size": matcher.get_enrolled_count(),
    }


@app.post("/match")
async def match_probe(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """
    Evaluates a probe image against the gallery using Discern's open-set decision engine.
    Accepts multipart file upload OR base64 payload (from webcam frame).
    """
    global matcher, extractor
    if not matcher or not extractor:
        raise HTTPException(status_code=500, detail="Matching engine not initialized")

    if matcher.get_enrolled_count() == 0:
        raise HTTPException(status_code=400, detail="Gallery is empty. Please enroll identities first.")

    # Obtain PIL Image
    pil_img: Optional[Image.Image] = None
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")
        image_b64 = body.get("image_base64")
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image provided (upload a file or send image_base64)")
        try:
            header_split = image_b64.split(",")
            base64_data = header_split[-1]
            decoded = base64.b64decode(base64_data)
            pil_img = Image.open(io.BytesIO(decoded)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.")
    elif file is not None:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
        if len(contents) > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Probe image exceeds maximum allowed size (15MB).")
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid or readable image.")
    else:
        raise HTTPException(status_code=400, detail="No image provided (upload a file or send image_base64)")

    # Check for empty / zero-variance image
    img_arr = np.array(pil_img)
    if float(np.std(img_arr)) < 3.0:
        raise HTTPException(
            status_code=400,
            detail="Probe image is blank or contains no subject (insufficient visual variance)."
        )

    t_start = time.perf_counter()
    try:
        query_emb = extractor.extract(pil_img)
        match_res = matcher.match(query_emb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching pipeline failed: {str(e)}")
    t_end = time.perf_counter()
    elapsed_ms = round((t_end - t_start) * 1000.0, 1)

    res_dict = match_res.to_dict()
    res_dict["inference_time_ms"] = elapsed_ms

    # Baseline plain cosine threshold comparison (standard fixed threshold = 0.70)
    top_cand = res_dict["top_candidates"][0] if res_dict.get("top_candidates") else None
    baseline_th = 0.70
    baseline_accepted = (res_dict["raw_similarity"] >= baseline_th) and (top_cand is not None)
    is_false_accept = baseline_accepted and (match_res.decision == "UNKNOWN")

    res_dict["baseline_comparison"] = {
        "decision": "ACCEPTED" if baseline_accepted else "UNKNOWN",
        "predicted_id": top_cand["identity_id"] if baseline_accepted else None,
        "predicted_name": top_cand["name"] if baseline_accepted else None,
        "similarity": res_dict["raw_similarity"],
        "threshold": baseline_th,
        "is_false_accept": is_false_accept,
        "explanation": (
            f"Plain cosine accepts at similarity {res_dict['raw_similarity']:.3f} >= {baseline_th:.2f}, "
            "failing to detect competitor ambiguity!"
            if is_false_accept
            else (
                f"Accepted under threshold {baseline_th:.2f}"
                if baseline_accepted
                else f"Rejected: similarity {res_dict['raw_similarity']:.3f} < {baseline_th:.2f}"
            )
        ),
    }

    # Enrich top candidates with sample thumbnail URLs
    for cand in res_dict.get("top_candidates", []):
        cid = cand["identity_id"]
        proto = matcher.prototypes.get(cid)
        if proto and proto.image_paths:
            cand["thumbnail_url"] = f"/static/{os.path.relpath(proto.image_paths[0], PROJECT_ROOT).replace('\\', '/')}"
        else:
            cand["thumbnail_url"] = None

    return res_dict


@app.get("/cross-dataset")
def get_cross_dataset():
    """Returns zero-shot cross-dataset evaluation results across benchmark repositories."""
    cross_path = os.path.join(RESULTS_DIR, "cross_dataset.json")
    if not os.path.isfile(cross_path):
        return {
            "status": "empty",
            "message": "Run scripts/evaluate_cross_dataset.py to populate cross-dataset benchmarks",
            "datasets": {},
        }
    with open(cross_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "status": "ready",
        "datasets": data,
    }


@app.get("/robustness")
def get_robustness():
    """Returns perturbation stress-test benchmarks across 50 random queries."""
    rob_path = os.path.join(RESULTS_DIR, "robustness.json")
    if not os.path.isfile(rob_path):
        return {
            "status": "empty",
            "message": "Run scripts/robustness_check.py to populate robustness benchmarks",
            "perturbations": {},
        }
    with open(rob_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "status": "ready",
        "perturbations": data,
    }


@app.post("/demo/scenario")
def run_demo_scenario():
    """
    Executes the 3-step live demo scenario on real benchmark probes:
    Step 1: Genuine probe -> ACCEPTED
    Step 2: Distant visitor probe -> UNKNOWN (low similarity)
    Step 3: Look-alike impostor probe -> UNKNOWN (look-alike safety barrier refused)
    """
    global matcher, extractor
    if not matcher or not extractor:
        raise HTTPException(status_code=500, detail="Matching engine not initialized")

    demo_steps = [
        {
            "step": 1,
            "title": "Step 1: Genuine Staff Acceptance",
            "category": "genuine",
            "probe_path": "data/sample_market1501/query/0026_c4s1_002604_00.jpg",
            "probe_url": "/static/data/sample_market1501/query/0026_c4s1_002604_00.jpg",
            "expected": "ACCEPTED",
            "caption": "Genuine staff member (Identity 26) correctly recognized with high similarity (0.853) and distinct competitive margin (0.231).",
        },
        {
            "step": 2,
            "title": "Step 2: Unenrolled Distant Visitor Rejection",
            "category": "distant_visitor",
            "probe_path": "data/sample_market1501/bounding_box_test/0004_c3s1_000403_00.jpg",
            "probe_url": "/static/data/sample_market1501/bounding_box_test/0004_c3s1_000403_00.jpg",
            "expected": "UNKNOWN",
            "caption": "Unenrolled visitor cleanly rejected as UNKNOWN due to similarity (0.588) falling below the operating threshold.",
        },
        {
            "step": 3,
            "title": "Step 3: Uniform Look-Alike Impostor Refusal (The Core Novelty)",
            "category": "lookalike_impostor",
            "probe_path": "data/sample_market1501/bounding_box_test/0030_c3s1_003003_00.jpg",
            "probe_url": "/static/data/sample_market1501/bounding_box_test/0030_c3s1_003003_00.jpg",
            "expected": "UNKNOWN",
            "caption": "Look-alike impostor safely refused. Plain cosine produces a dangerous False Accept (similarity 0.740 >= 0.70), while Discern activates the competitive margin barrier (margin 0.014 < 0.067) to safely reject admittance.",
        },
    ]

    results = []
    for s in demo_steps:
        full_p = os.path.join(PROJECT_ROOT, s["probe_path"])
        if not os.path.isfile(full_p):
            continue
        img = Image.open(full_p).convert("RGB")
        emb = extractor.extract(img)
        match_res = matcher.match(emb)
        res_d = match_res.to_dict()

        # Baseline comparison
        top_cand = res_d["top_candidates"][0] if res_d.get("top_candidates") else None
        baseline_accepted = (res_d["raw_similarity"] >= 0.70) and (top_cand is not None)
        is_false_accept = baseline_accepted and (match_res.decision == "UNKNOWN")

        # Candidate thumbnails
        for cand in res_d.get("top_candidates", []):
            cid = cand["identity_id"]
            proto = matcher.prototypes.get(cid)
            cand["thumbnail_url"] = (
                f"/static/{os.path.relpath(proto.image_paths[0], PROJECT_ROOT).replace('\\', '/')}"
                if proto and proto.image_paths
                else None
            )

        results.append({
            "step": s["step"],
            "title": s["title"],
            "category": s["category"],
            "probe_url": s["probe_url"],
            "caption": s["caption"],
            "expected": s["expected"],
            "discern_verdict": match_res.decision,
            "baseline_verdict": "ACCEPTED" if baseline_accepted else "UNKNOWN",
            "is_false_accept_prevented": is_false_accept,
            "confidence": res_d["calibrated_confidence"],
            "similarity": res_d["raw_similarity"],
            "competitor_similarity": res_d["competitor_similarity"],
            "margin": res_d["margin"],
            "margin_delta": res_d["operating_point"]["margin_delta"],
            "threshold_tau": res_d["operating_point"]["threshold_tau"],
            "human_reason": res_d["human_reason"],
            "top_candidates": res_d.get("top_candidates", [])[:2],
        })

    return {
        "status": "success",
        "scenario_name": "Open-Set Look-Alike Security Audit",
        "steps": results,
    }


@app.put("/operating-point")
def update_operating_point(update: OperatingPointUpdate):
    """Updates the target false-accept rate (alpha) and adjusts operating thresholds live."""
    global matcher
    if not matcher:
        raise HTTPException(status_code=500, detail="Matching engine not initialized")

    op = matcher.set_operating_point(update.alpha)
    return {
        "status": "success",
        "operating_point": op,
    }


@app.get("/metrics")
def get_metrics():
    """Returns headline metrics from real evaluation run."""
    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if not os.path.isfile(eval_path):
        return {
            "status": "empty",
            "message": "Run evaluation to populate metrics",
            "headline_metrics": None,
            "metadata": None,
        }

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "ready",
        "headline_metrics": data.get("headline_metrics"),
        "metadata": data.get("metadata"),
        "latency_and_parameters": data.get("latency_and_parameters"),
        "analysis_paragraph": data.get("analysis_paragraph"),
    }


@app.get("/roc")
def get_roc():
    """Returns ROC curve points for frontend Recharts plotting."""
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.json")
    if not os.path.isfile(roc_path):
        return {
            "status": "empty",
            "message": "Run evaluation to populate ROC curves",
            "roc_curves": None,
        }

    with open(roc_path, "r", encoding="utf-8") as f:
        roc_data = json.load(f)

    return {
        "status": "ready",
        "roc_curves": roc_data,
        "chart_png_url": "/static/results/roc_chart.png",
    }


@app.get("/ablation")
def get_ablation():
    """Returns step-by-step ablation study table."""
    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if not os.path.isfile(eval_path):
        return {
            "status": "empty",
            "message": "Run evaluation to populate ablation table",
            "ablation_study": [],
        }

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "ready",
        "ablation_study": data.get("ablation_study", []),
    }


@app.get("/lookalikes")
def get_lookalikes():
    """Returns curated look-alike pairs and clusters with representative image URLs."""
    lowvar_path = os.path.join(RESULTS_DIR, "lowvar_subset.json")
    if not os.path.isfile(lowvar_path):
        return {
            "status": "empty",
            "message": "Run prepare_data.py to populate look-alike clusters",
            "pairs": [],
            "clusters": [],
        }

    with open(lowvar_path, "r", encoding="utf-8") as f:
        lowvar = json.load(f)

    split_path = os.path.join(RESULTS_DIR, "open_set_split.json")
    id_to_img = {}
    if os.path.isfile(split_path):
        with open(split_path, "r", encoding="utf-8") as sf:
            sdata = json.load(sf)
            all_samples = (
                sdata.get("gallery_samples", [])
                + sdata.get("genuine_probe_samples", [])
                + sdata.get("impostor_probe_samples", [])
            )
            for s in all_samples:
                pid = s["identity_id"]
                if pid not in id_to_img:
                    id_to_img[pid] = f"/static/{os.path.relpath(s['image_path'], PROJECT_ROOT).replace('\\', '/')}"

    if matcher:
        for pid, proto in matcher.prototypes.items():
            if proto.image_paths and pid not in id_to_img:
                id_to_img[pid] = f"/static/{os.path.relpath(proto.image_paths[0], PROJECT_ROOT).replace('\\', '/')}"

    pairs = lowvar.get("top_lookalike_pairs", [])
    for p in pairs:
        p["image_url_a"] = id_to_img.get(p["identity_a"])
        p["image_url_b"] = id_to_img.get(p["identity_b"])

    return {
        "status": "ready",
        "metadata": lowvar.get("metadata", {}),
        "tightest_clusters": lowvar.get("tightest_clusters", []),
        "clusters": lowvar.get("tightest_clusters", []),
        "pairs": pairs,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
