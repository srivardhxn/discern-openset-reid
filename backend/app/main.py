"""
FastAPI backend for Discern Open-Set Re-ID (PS-1 ONNX model bundle).

Wraps models/discern_model/discern_inference.py (ONNX, GPU if available).
All decision math is delegated to Discern.identify() -- no reimplementation.

Endpoints
---------
POST   /api/enroll            identity: str, images: UploadFile[]
POST   /api/identify          image: UploadFile, op: strict|balanced|lenient
DELETE /api/identity/{id}
GET    /api/gallery
GET    /api/config            decision_config.json
GET    /api/lookalikes        lookalike_explorer.json (+ full image URLs)
GET    /api/metrics           reports/eval_report.json
GET    /api/metrics/images    URLs for ROC/ablation PNGs
POST   /api/demo/load         loads demo_manifest.json + demo_embeddings.npz
GET    /api/demo/status
POST   /api/demo/smoke-test   runs probes, returns accepted/FA counts
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]       # …/discern/
MODEL_DIR    = PROJECT_ROOT / "model"
DEMO_DIR     = MODEL_DIR / "demo"
SAMPLES_DIR  = MODEL_DIR / "samples"
REPORTS_DIR  = MODEL_DIR / "reports"

# Inject model dir so Python can find discern_inference.py
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from discern_inference import Discern  # noqa: E402

# ---------------------------------------------------------------------------
# Boot model (GPU auto-selected)
# ---------------------------------------------------------------------------
_model = Discern(str(MODEL_DIR))
print("[Discern] Providers:", _model.sess.get_providers())

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Discern ONNX Re-ID API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static/samples",     StaticFiles(directory=str(SAMPLES_DIR)),          name="samples")
app.mount("/static/demo/images", StaticFiles(directory=str(DEMO_DIR / "images")),  name="demo_images")
app.mount("/static/reports",     StaticFiles(directory=str(REPORTS_DIR)),          name="reports")

# ---------------------------------------------------------------------------
# Demo state
# ---------------------------------------------------------------------------
_demo_loaded: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pil(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "File is not a readable image.")


def _validate(img: Image.Image, name: str = "image") -> None:
    if float(np.std(np.array(img))) < 3.0:
        raise HTTPException(400, f"{name} is blank (insufficient visual variance).")


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "model": _model.cfg["model"]["name"],
        "providers": _model.sess.get_providers(),
        "enrolled": len(_model.rows),
        "default_op": _model.cfg["default_operating_point"],
    }


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------
@app.get("/api/gallery")
def get_gallery():
    """Return summary of all enrolled identities."""
    return [
        {"identity": name, "num_embeddings": len(rows)}
        for name, rows in _model.rows.items()
    ]


# ---------------------------------------------------------------------------
# Enroll
# ---------------------------------------------------------------------------
@app.post("/api/enroll")
async def enroll(
    identity: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """Enroll one or more person crops under a named identity."""
    identity = identity.strip()
    if not identity:
        raise HTTPException(400, "Identity name cannot be empty.")
    if not images:
        raise HTTPException(400, "At least one image is required.")

    pils: List[Image.Image] = []
    for up in images:
        img = _pil(up)
        _validate(img, up.filename or "image")
        pils.append(img)

    _model.enroll(identity, pils)
    return {
        "status": "enrolled",
        "identity": identity,
        "images_added": len(pils),
        "total_enrolled": len(_model.rows),
    }


# ---------------------------------------------------------------------------
# Identify
# ---------------------------------------------------------------------------
@app.post("/api/identify")
async def identify(
    image: UploadFile = File(...),
    op: Optional[str] = Form(None),
):
    """
    Identify a probe against the enrolled gallery.
    op: strict | balanced | lenient  (default from decision_config.json)
    Always returns decision='reject'/display_decision='UNKNOWN' when rejected.
    Never forces a match.
    """
    if op and op not in ("strict", "balanced", "lenient"):
        raise HTTPException(400, "op must be strict, balanced, or lenient")
    if not _model.rows:
        raise HTTPException(400, "Gallery empty. Load demo or enroll identities.")

    img = _pil(image)
    _validate(img)

    t0 = time.perf_counter()
    result = _model.identify(img, op=op)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if result["decision"] != "accept":
        result["identity"] = "UNKNOWN"
    result["display_decision"] = "ACCEPTED" if result["decision"] == "accept" else "UNKNOWN"
    return result


# ---------------------------------------------------------------------------
# Delete identity
# ---------------------------------------------------------------------------
@app.delete("/api/identity/{identity_id}")
def delete_identity(identity_id: str):
    """Remove an identity from the gallery (rebuilds embedding matrix)."""
    if identity_id not in _model.rows:
        raise HTTPException(404, f"Identity {identity_id!r} not found in gallery.")
    _model.remove(identity_id)
    return {"status": "deleted", "identity": identity_id, "remaining": len(_model.rows)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@app.get("/api/config")
def get_config():
    """Return the full decision_config.json as JSON."""
    return _model.cfg


# ---------------------------------------------------------------------------
# Lookalikes
# ---------------------------------------------------------------------------
@app.get("/api/lookalikes")
def get_lookalikes():
    """Return lookalike_explorer.json with full image URLs attached."""
    p = MODEL_DIR / "lookalike_explorer.json"
    if not p.exists():
        raise HTTPException(404, "lookalike_explorer.json not found.")
    data = json.loads(p.read_text())
    for pair in data.get("pairs", []):
        for field in ("image_a", "image_b", "genuine_image_1", "genuine_image_2"):
            raw: Optional[str] = pair.get(field)
            if raw:
                pair[f"{field}_url"] = "/static/samples/" + Path(raw).name
    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/api/metrics")
def get_metrics():
    """Return reports/eval_report.json."""
    p = REPORTS_DIR / "eval_report.json"
    if not p.exists():
        raise HTTPException(404, "eval_report.json not found.")
    return json.loads(p.read_text())


@app.get("/api/metrics/images")
def get_metric_images():
    """Return static URLs for report PNG images."""
    fmap = {
        "roc_ablation":    "open_set_roc_ablation.png",
        "calibration":     "calibration_reliability.png",
        "training_curves": "training_curves.png",
        "lookalike_pairs": "lookalike_pairs.png",
    }
    return {
        k: "/static/reports/" + f
        for k, f in fmap.items()
        if (REPORTS_DIR / f).exists()
    }


@app.get("/api/metrics/ablation-inference")
def ablation_inference():
    p = REPORTS_DIR / "ablation_inference.md"
    return {"content": p.read_text() if p.exists() else ""}


@app.get("/api/metrics/ablation-training")
def ablation_training():
    p = REPORTS_DIR / "ablation_training.md"
    return {"content": p.read_text() if p.exists() else ""}


# ---------------------------------------------------------------------------
# Demo load
# ---------------------------------------------------------------------------
@app.post("/api/demo/load")
def demo_load():
    """
    Load the bundled demo gallery.
    Uses pre-computed demo_embeddings.npz (fast) if available,
    otherwise embeds gallery images on the fly.
    Clears any existing gallery first.
    """
    global _demo_loaded
    manifest_p = DEMO_DIR / "demo_manifest.json"
    emb_p      = DEMO_DIR / "demo_embeddings.npz"
    if not manifest_p.exists():
        raise HTTPException(404, "demo_manifest.json not found.")

    manifest = json.loads(manifest_p.read_text())

    # Reset gallery
    _model.rows.clear()
    _model.emb = np.zeros((0, _model.cfg["model"]["embedding_dim"]), np.float32)

    if emb_p.exists():
        npz = np.load(str(emb_p))
        if "gallery" in npz and len(manifest.get("gallery", [])) == len(npz["gallery"]):
            for i, entry in enumerate(manifest["gallery"]):
                emb = npz["gallery"][i : i + 1].astype(np.float32)
                _model.add_embeddings(entry["identity"], emb)
            via = "pre-computed embeddings"
        else:
            for key in npz.files:
                emb = npz[key].astype(np.float32)
                if emb.ndim == 1:
                    emb = emb[np.newaxis, :]
                _model.add_embeddings(key, emb)
            via = "pre-computed embeddings"
    else:
        for entry in manifest["gallery"]:
            img_path = DEMO_DIR / entry["file"]
            if img_path.exists():
                _model.enroll(entry["identity"], [Image.open(str(img_path)).convert("RGB")])
        via = "on-the-fly embedding"

    _demo_loaded = True
    return {
        "status": "loaded",
        "enrolled_identities": len(_model.rows),
        "method": via,
        "gallery_entries": len(manifest.get("gallery", [])),
        "probe_entries":   len(manifest.get("probes", [])),
    }


@app.get("/api/demo/status")
def demo_status():
    return {
        "demo_loaded": _demo_loaded,
        "enrolled_identities": len(_model.rows),
        "identities": list(_model.rows.keys()),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
@app.post("/api/demo/smoke-test")
def smoke_test():
    """
    Run demo probes through the loaded gallery.
    Returns known-accepted and look-alike-false-accept counts.
    """
    manifest_p = DEMO_DIR / "demo_manifest.json"
    if not manifest_p.exists():
        raise HTTPException(404, "demo_manifest.json not found.")
    if not _model.rows:
        raise HTTPException(400, "Gallery empty. Call POST /api/demo/load first.")

    manifest = json.loads(manifest_p.read_text())
    ok = tot = fa = unk = 0
    results = []

    for p in manifest.get("probes", []):
        img_path = DEMO_DIR / p["file"]
        if not img_path.exists():
            continue
        img = Image.open(str(img_path)).convert("RGB")
        r = _model.identify(img)
        truth    = p.get("truth", "")
        expected = p.get("expected", "")
        if truth == "enrolled":
            tot += 1
            ok  += int(r["identity"] == expected)
        else:
            unk += 1
            fa  += int(r["decision"] == "accept")
        results.append({
            "file": p["file"], "truth": truth, "expected": expected,
            "decision": r["decision"], "identity": r.get("identity"),
            "confidence": r["confidence"],
        })

    return {
        "known_accepted": ok,
        "known_total": tot,
        "lookalike_false_accepts": fa,
        "lookalike_total": unk,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Legacy compatibility shims (keep old frontend working during transition)
# ---------------------------------------------------------------------------
@app.get("/gallery")
def legacy_gallery():
    return get_gallery()


@app.delete("/identity/{identity_id}")
def legacy_delete(identity_id: str):
    return delete_identity(identity_id)


@app.get("/metrics")
def legacy_metrics():
    try:
        return get_metrics()
    except HTTPException:
        return {"status": "empty"}


@app.get("/lookalikes")
def legacy_lookalikes():
    try:
        data = get_lookalikes()
        pairs = [
            {
                "identity_a":          pair.get("id_a"),
                "identity_b":          pair.get("id_b"),
                "clothing_similarity": pair.get("colour_similarity", 0.0),
                "impostor_similarity": pair.get("impostor_similarity", 0.0),
                "impostor_confidence": pair.get("impostor_confidence", 0.0),
                "genuine_similarity":  pair.get("genuine_similarity", 0.0),
                "image_url_a":         pair.get("image_a_url"),
                "image_url_b":         pair.get("image_b_url"),
                "genuine_image_1_url": pair.get("genuine_image_1_url"),
                "genuine_image_2_url": pair.get("genuine_image_2_url"),
                "rank":                pair.get("rank", 0),
            }
            for pair in data.get("pairs", [])
        ]
        return {
            "status": "ready",
            "pairs": pairs,
            "metadata": {"total_pairs": len(pairs), "description": data.get("description", "")},
        }
    except HTTPException:
        return {"status": "empty", "pairs": [], "metadata": {}}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
