# Discern model bundle — integration contract

Open-set person re-identification (Zed Digital PS-1). Everything the app needs is in this folder.

| file | purpose |
|---|---|
| `discern_embedder.onnx` | **the model.** input `images` float32 `[N,3,288,144]` RGB in 0..1 -> output `embedding` `[N,2048]` L2-normalised (ImageNet normalisation is inside) |
| `decision_config.json` | preprocessing, TTA rule, aggregation, calibrators and operating-point thresholds — **single source of truth** |
| `discern_inference.py` | reference backend (`Discern.embed / enroll / identify`), onnxruntime only |
| `lookalike_explorer.json` + `samples/` | 60 most confusable different-person pairs (+ a genuine pair for contrast) for the Look-alike Explorer page |
| `demo/` | pre-computed demo gallery/probes (`demo_manifest.json`, `demo_embeddings.npz`, images) so the UI works with zero setup |
| `reports/` | `eval_report.json`, `ablation_inference.md`, `ablation_training.md`, ROC / calibration / training PNGs |
| `discern_reid_r50.pt`, `discern_embedder.torchscript.pt` | PyTorch weights / TorchScript (optional) |

## Pipeline (what the backend must do)
1. Person crop (RGB) -> resize/stretch to **144x288** (bilinear) -> float 0..1 -> NCHW.
2. `e = normalize(onnx(x) + onnx(flip_horizontal(x)))`  (flip-TTA, used for enrolment **and** probes).
3. Gallery = list of `(identity, embedding)`; identity score = mean of top-k cosine sims (`aggregation` in config: **max**).
4. Features: `s1` (best identity), `margin = s1 - runner_up`, `z = (s1 - mean(S)) / std(S)`.
5. `confidence = sigmoid(coef · ((f-mean)/scale) + intercept)` using calibrator `s1_margin_z` (>=10 identities), else `s1_margin`, else `s1`.
6. Accept iff `confidence > threshold[operating_point]`, otherwise answer **UNKNOWN** (never force a match).

## Operating points (calibrator `s1_margin_z`, pooled Market-1501 open-set protocol)
| name | threshold | false-accept | look-alike (LV90) false-accept | true-accept |
|---|---|---|---|---|
| strict | 0.998 | 0.10% | 0.82% | 8.8% |
| balanced | 0.963 | 0.99% | 5.46% | 51.4% |
| lenient | 0.651 | 5.00% | 13.20% | 82.7% |

Closed-set Market-1501: mAP **86.3**, Rank-1 **94.2** (flip-TTA, single query, no re-ranking).
Calibration on held-out probes: ECE **0.014**.

## Suggested REST surface
`POST /api/enroll` (identity, images[]) · `POST /api/identify` (image, op?) · `DELETE /api/identity/{id}` · `GET /api/config` (decision_config.json)
· `GET /api/lookalikes` (lookalike_explorer.json) · `GET /api/metrics` (reports/eval_report.json) · `POST /api/demo/load`.
Note: thresholds were fitted with ~375 enrolled identities; keep an operating-point selector (strict / balanced / lenient) in the UI.
