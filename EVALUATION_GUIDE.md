# DISCERN: Evaluation & Project Execution Guide
**Open-Set Person Re-Identification Under Low Inter-Class Appearance Variance**

> **Author / Student Submission**  
> **Repository:** [https://github.com/srivardhxn/discern-openset-reid](https://github.com/srivardhxn/discern-openset-reid)  
> **Evaluation Package Version:** 1.0.0 (Windows / Linux / macOS Compatible)

---

## 1. Executive Summary & What is Happening

### 1.1 The Real-World Problem
Traditional **Person Re-Identification (Re-ID)** models assume a **closed-set** environment: they assume the query person *must* already exist in the gallery database, forcing a nearest-neighbour match.

In real-world security, medical, and industrial facilities, two major problems occur:
1. **Low Inter-Class Appearance Variance (Uniformed Personnel / Look-Alikes):**
   Security officers, hospital staff, factory workers, or warehouse employees wear **identical uniforms** or similar clothing. Standard models heavily rely on clothing colour and texture, causing distinct individuals in matching clothes to appear identical in embedding space.
2. **Open-Set Reality (Strangers & Impostors):**
   Most people appearing on camera are unregistered strangers. Standard closed-set systems suffer catastrophic **False Accepts (FAR)** by forcing high-confidence matches on unknown individuals.

### 1.2 How Discern Works Under the Hood
**Discern** is an open-set Person Re-ID system engineered to suppress false accepts among look-alikes while maintaining high recall on enrolled subjects.

The complete pipeline works as follows:

```
[ Person Crop Image (288x144) ]
               │
               ▼
┌────────────────────────────────────────┐
│ Preprocessing & Normalization          │
│ • Resize to 288x144 (NCHW)             │
│ • Internal ImageNet standardization    │
└────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ ResNet-50 Feature Embedder             │
│ • Last Stride = 1 (preserves spatial)  │
│ • Generalized Mean (GeM) Pooling       │
│ • Batch Normalization Neck (BNNeck)    │
│ • Horizontal Flip Test-Time Aug (TTA)  │
│ ➔ Produces 2048-d L2-normalized vector │
└────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ Multi-Shot Gallery Aggregation         │
│ • Computes cosine similarity to gallery│
│ • Aggregates multi-image identities    │
│   (max / top-k mean similarity)        │
└────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ Triple-Feature Extraction              │
│ 1. s1: Top-candidate similarity        │
│ 2. margin (m = s1 - s2): Runner-up gap │
│ 3. z-score (z = (s1 - μ) / σ):         │
│    Query-adaptive gallery dispersion   │
└────────────────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│ Calibrated Decision Engine             │
│ • Logistic calibrator outputs posterior│
│   confidence score ∈ [0.0, 1.0]        │
│ • Compares with Operating Point:       │
│   - Strict   (τ = 0.998, FAR: 0.10%)   │
│   - Balanced (τ = 0.963, FAR: 0.99%)   │
│   - Lenient  (τ = 0.651, FAR: 5.00%)   │
└────────────────────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
[ ACCEPT: Enrolled ]  [ REJECT: UNKNOWN ]
```

1. **Feature Extraction (`discern_embedder.onnx`):**
   - ResNet-50 backbone modified with last-stride=1 to preserve fine-grained spatial feature maps.
   - **GeM Pooling (Generalized Mean)** captures localized identity cues (badges, facial contours, posture) better than standard average pooling.
   - Outputs a **2048-dimensional unit vector ($L_2$-normalized)**.
   - Employs **Horizontal Flip Test-Time Augmentation (TTA)**: embeds both original and flipped images ($e = \text{norm}(e_{\text{orig}} + e_{\text{flip}})$), eliminating orientation bias.
2. **Multi-Shot Gallery Aggregation:**
   - When an identity has multiple enrolled photos, the similarity score is aggregated across viewpoints.
3. **Triple-Feature Calibrated Scoring ($s_1, \text{margin}, z$):**
   - **$s_1$ (Top-1 Similarity):** Cosine similarity to the best-matching identity.
   - **Margin ($m = s_1 - s_2$):** Difference between the top candidate and the runner-up candidate. In look-alike scenarios (same uniform), multiple identities score high; a small margin signals identity ambiguity and prevents a false accept.
   - **Query Z-Score ($z = \frac{s_1 - \mu}{\sigma}$):** Measures how prominently the top match stands out against the general gallery background distribution.
4. **Calibrated Confidence & Operating Points:**
   - A calibrated logistic regressor converts $(s_1, m, z)$ into an absolute probability $P(\text{same-ID} \mid x)$.
   - Evaluates against the selected operating point:
     - **Strict (Threshold 0.998):** For high-security access control. Minimizes False Accept Rate to 0.10%.
     - **Balanced (Threshold 0.963 - Default):** Balances FAR (0.99%) with True Accept Rate (TAR 51.4%).
     - **Lenient (Threshold 0.651):** For forensic review (TAR 82.7%, FAR 5.0%).
   - If confidence is below the threshold, Discern rejects the subject as **`UNKNOWN`**.

---

## 2. System Requirements & Prerequisites

- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Version 3.10 or higher (`python --version`).
- **Node.js:** Version 18 or higher with npm (`node --version`).
- **Hardware:** Runs on standard CPU (ONNX Runtime is optimized for CPU; NVIDIA CUDA GPU is automatically utilized if present).

---

## 3. Quick Start: How to Run in 2 Minutes

### Method A: Automated PowerShell Scripts (Windows)

#### Step 1: Run Environment Setup
Open PowerShell inside the unzipped project folder and execute:
```powershell
.\setup.ps1
```
*What this script does:*
- Checks Python 3.10+ and Node.js 18+.
- Creates the local Python virtual environment (`.venv`).
- Installs backend dependencies from `requirements.txt`.
- Installs frontend packages in `frontend/` (`npm install`).
- Validates the ONNX model bundle.

#### Step 2: Start the Backend API Server
In a PowerShell window:
```powershell
.\run_api.ps1
```
*The backend API will start at:* **`http://localhost:8000`**  
*Interactive Swagger API documentation:* **`http://localhost:8000/docs`**

#### Step 3: Start the Frontend UI Dashboard
In a second PowerShell window:
```powershell
.\run_ui.ps1
```
*The web dashboard will launch at:* **`http://localhost:5173`**

---

### Method B: Manual Commands (Windows / Linux / macOS)

If you prefer running standard terminal commands or are on Linux/macOS:

#### Terminal 1 — Backend API
```bash
# 1. Create and activate virtual environment
python -m venv .venv

# Windows activation:
.\.venv\Scripts\Activate.ps1
# Linux/macOS activation:
# source .venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Start FastAPI server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 — Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 4. Instant CLI Verification (30-Second Smoke Test)

If you want to immediately verify that the model, inference pipeline, and decision rules work correctly without launching the web browser:

```powershell
# Run the automated PyTest suite:
.\.venv\Scripts\pytest.exe tests/test_model_bundle.py -v
```

**Expected Test Output:**
- `test_decision_config_structure`: PASSED (verifies 2048-d embedder, operating thresholds).
- `test_model_demo_replay`: PASSED (replays 40 gallery identities + 40 probes; confirms 40/40 enrolled probes accepted, 2/40 look-alike impostors false accepted as per calibrated balanced threshold).

You can also run the standalone smoke test against the running backend:
```powershell
python scripts/smoke_test.py
```

---

## 5. How to Evaluate the Web Dashboard (Evaluation Walkthrough)

Open your web browser at **`http://localhost:5173`**. The dashboard contains 5 dedicated evaluation tabs:

### 1. Overview Tab
- **System Architecture:** Visual summary of ResNet-50 + GeM + BNNeck + Margin calibration.
- **Benchmark Metrics:**
  - Closed-Set Market-1501: **Rank-1: 94.2%**, **mAP: 86.3%**.
  - Calibration Error (ECE): **0.014** (extremely well calibrated probabilities).
- **Inference & Training Ablation Tables:** Demonstrating the impact of Flip-TTA, runner-up margin, and z-score normalization.

### 2. Gallery & Enrollment Tab
- **Preload Demo Gallery:** Click the **"Load Bundled Demo Gallery"** button. This instantly enrolls 40 identities into memory with pre-calculated embeddings.
- **Custom Enrollment:** Upload one or more person crops (PNG/JPG), enter an Identity Name/ID, and click **Enroll**. The system computes 2048-d embeddings with flip-TTA and adds them to the live gallery.

### 3. Match / Identification Tab (Core Feature)
- **Select or Upload Probe:** Upload any test image or pick from the sample probe images in `model/demo/`.
- **Select Operating Point:**
  - `Strict` (0.998): Extremely tight rejection of strangers.
  - `Balanced` (0.963): Default production setting.
  - `Lenient` (0.651): Relaxed threshold.
- **Click "Identify":**
  - **If the person is enrolled:** Displays green **MATCH ACCEPTED** badge with Identity Name, Confidence %, Top-1 Similarity ($s_1$), Runner-up Margin ($m$), and Query Z-score ($z$).
  - **If the person is a stranger/look-alike:** Displays amber **UNKNOWN PERSON (REJECTED)** badge, showing that even though a visual look-alike exists in the gallery, the competitive margin and calibrated threshold correctly suppressed a false accept!

### 4. Look-Alikes Explorer Tab
- Displays the **60 hardest confusable pairs** from the dataset (people wearing identical red shirts, blue jackets, or worker outfits).
- Illustrates how naive cosine similarity fails by rating them as high similarity, whereas Discern's margin and calibration rule successfully differentiates them.

### 5. Evaluation & Reports Tab
- High-resolution evaluation plots:
  - **Open-Set ROC Curves:** Comparing baseline A0 vs full Discern A4 across low-variance splits (LV75, LV90).
  - **Calibration Reliability Diagram:** Verifying predicted confidence vs empirical accuracy.
  - **Look-Alike Visualizer:** Grid of sample confusable crops.

---

## 6. Project Directory Layout

```
discern/
├── EVALUATION_GUIDE.md               # Complete evaluation and architecture guide (this file)
├── HOW_TO_RUN_AND_EVALUATION_STEPS.txt # Plain text version for quick reading
├── README.md                          # Repository overview and benchmark documentation
├── requirements.txt                   # Python dependencies (fastapi, onnxruntime, pillow, etc.)
├── setup.ps1                          # 1-click Windows setup script
├── run_api.ps1                        # 1-click backend launcher (port 8000)
├── run_ui.ps1                         # 1-click frontend launcher (port 5173)
│
├── backend/                           # FastAPI backend
│   └── app/
│       └── main.py                    # REST API routes (/api/enroll, /api/identify, etc.)
│
├── frontend/                          # React + TypeScript + Tailwind CSS dashboard
│   ├── src/
│   │   ├── pages/                     # Overview, Enroll, Match, LookAlikes, Evaluation
│   │   ├── api.ts                     # REST client for backend communication
│   │   └── App.tsx                    # Main app navigation & layout
│   └── package.json                   # Node.js dependencies
│
├── model/                             # Core Discern model bundle
│   ├── discern_embedder.onnx          # Exported ResNet-50 ONNX embedder (2048-d)
│   ├── decision_config.json           # Calibrator weights, feature specs, operating points
│   ├── discern_inference.py           # Core ONNX Runtime inference & decision class
│   ├── lookalike_explorer.json        # 60 hardest confusable person pairs
│   ├── demo/                          # 40 gallery identities & 40 probe test crops
│   ├── reports/                       # Evaluation charts (ROC, calibration, training curves)
│   └── samples/                       # Sample crops for visual explorer
│
├── Notebook Scripts/
│   └── visionmodel.ipynb              # Complete research training & evaluation notebook
│
├── scripts/
│   └── smoke_test.py                  # Standalone CLI verification test
│
└── tests/
    └── test_model_bundle.py           # Automated PyTest verification suite
```

---

## 7. Troubleshooting & FAQs

### Q1: PowerShell says "running scripts is disabled on this system"
Run this command in PowerShell to temporarily allow local script execution:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Q2: Port 8000 or Port 5173 is already in use
- To run the backend on a different port (e.g. 8001):
  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8001
  ```
  *(Note: Update `frontend/src/api.ts` base URL if changed).*
- To run the frontend on a different port:
  ```bash
  npm run dev -- --port 5174
  ```

### Q3: Does this project require an NVIDIA GPU?
No. The ONNX Runtime embedder runs efficiently on any standard modern CPU (~150ms per probe with flip-TTA). If an NVIDIA GPU with CUDA is detected, ONNX Runtime automatically uses `CUDAExecutionProvider` for accelerated sub-30ms inference.

### Q4: Where can I view the raw training and experimentation notebook?
The complete training pipeline, loss formulation (Batch-Hard Triplet + Look-Alike Mining + Cross-Entropy with Label Smoothing), and ablation studies are preserved in:
`Notebook Scripts/visionmodel.ipynb`.

---
*For any questions or clarification during evaluation, please refer to the GitHub repository at https://github.com/srivardhxn/discern-openset-reid.*
