# Discern: 3-Minute Live Evaluation & Demo Script

This script walks through demonstrating **Discern**, an open-set person re-identification system engineered to minimize false accepts when subjects wear identical uniforms (low inter-class appearance variance).

---

## ⏱️ Pre-Flight Verification (30 Seconds)
Ensure the backend API and frontend dev server are running:
```powershell
# Terminal 1: Launch Backend API
.\run_api.ps1

# Terminal 2: Launch Frontend UI
.\run_ui.ps1
```
Open your browser to: **`http://localhost:5173`** (or `http://localhost:8000/docs` for OpenAPI).

---

## 🎯 Minute 1: The Core Problem & Architectural Breakthrough (Overview Page)

1. **Navigate to the "Overview" Tab**:
   - Point out the headline summary: *Traditional Re-ID systems rely overwhelmingly on dominant color silhouettes. In environments with shared uniforms (police, security guards, hospital staff), subjects share 90%+ appearance similarity.*
2. **Review the Real Evaluation Headline Metrics**:
   - **TAR @ FAR = 1.0%**: Notice how Discern achieves **100.0%** True Accept Rate at strict 1% False Accept Rate on the curated low-variance uniform subset.
   - **Detection & Identification Rate (DIR @ 1% FAR)**: Shows **100.0%** correct rank-1 identity attribution.
   - **CPU Inference Latency**: **~93 ms/image** (over 10 FPS on standard commodity CPU without GPU requirement).
   - **Model Footprint**: **0.61M parameters** (compact multi-scale OSNet backbone).
3. **Walk Through the 5-Stage Pipeline Diagram**:
   - `Input Image` ➔ `Horizontal 3-Stripe Pooling (512-d)` ➔ `Gallery-Adaptive Whitening` ➔ `Dual-Barrier Test (s₁ ≥ τᵢ & s₁ - s₂ ≥ δ)` ➔ `Calibrated Decision`.

---

## 🎯 Minute 2: Live Dual-Barrier Rejection Demonstration (Match & Verify Page)

1. **Navigate to the "Match & Verify" Tab**:
   - Highlight the **Target False Accept Rate (α) Slider** in the top bar (e.g. set to `1.00%` or strict `0.10%`).
2. **Demonstration A — Genuine Verification**:
   - Click the green **"✔ Genuine Probe"** button.
   - **Result Card**: Displays **`ACCEPTED as Identity 1`** with high calibrated confidence (~98%).
   - Note the **Top Similarity ($s_1$)**, **Competitor Similarity ($s_2$)**, and positive safety **Margin ($s_1 - s_2 \ge \delta$)**.
3. **Demonstration B — The Critical Look-Alike Impostor Test (The Novelty)**:
   - Click the amber **"⚠ Look-Alike Impostor"** button.
   - Point out the difference:
     - Under a standard cosine threshold (which would accept at similarity > 0.60 or 0.85), this look-alike would be **falsely accepted**!
     - Discern triggers **`UNKNOWN (REJECTED)`**.
     - Point out the human-readable explanation:
       `"Rejected: look-alike ambiguity. Too close to Identity 28 (margin 0.024 is below safety threshold 0.050)"`.
   - **Drag the Alpha Slider**: Show how adjusting $\alpha$ live recalibrates the operating threshold $\tau$ dynamically.

---

## 🎯 Minute 3: Scientific Rigor & Appearance Twins (Evaluation & Explorer)

1. **Navigate to the "Evaluation & Ablations" Tab**:
   - **Interactive ROC Curves**: Compare the green curve (*Discern on Low-Variance Uniforms*) against the dashed red baseline (*Plain Cosine*). Point out the vertical reference line at the target 1% FAR operating point.
   - **Step-by-Step Component Ablation Table**:
     - Walk down the rows:
       1. *Baseline (Plain Cosine)*
       2. *+ Margin Loss (ArcFace)*
       3. *+ Look-Alike PK Mining*
       4. *+ Horizontal Stripe Features*
       5. *+ Gallery Whitening (suppresses uniform covariance)*
       6. *+ Adaptive Threshold & Margin Test*
       7. *+ Calibration (Full Discern)*
     - Note the highlighted bold green numbers showing the progression to peak TAR@1% and AUROC.
2. **Navigate to the "Look-Alike Explorer" Tab**:
   - Inspect the **Side-by-Side Appearance Twins**:
     - See Identity 31 vs Identity 28 (88.7% clothing similarity) and Identity 25 vs Identity 1 (87.5% clothing similarity).
     - Point out the explanation card: *Discern estimates covariance over enrolled uniforms to de-correlate the shared fabric color, focusing decision boundaries on fine-grained torso badges, collar structures, and shoe geometry.*
3. **Wrap-up**:
   - Reiterate that all numbers, tables, curves, and look-alikes are driven by genuine evaluation result files produced from the open-set protocol.
