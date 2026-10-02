# Person Re-Identification Datasets Guide

Discern supports zero-shot evaluation across multiple public Re-ID benchmarks. Place downloaded datasets into the directory layouts specified below.

---

## 1. Market-1501 (Default Benchmark & Pre-Packaged Sample)
- **Source**: [Zheng et al., ICCV 2015](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- **Directory Layout**:
  ```text
  data/Market-1501-v15.09.15/  (or data/sample_market1501/)
  ├── bounding_box_train/       # 12,936 training images
  ├── bounding_box_test/        # 19,732 gallery images
  └── query/                    # 3,368 probe images
  ```
- **File Naming Format**: `[pid]_c[cam]s[seq]_[frame]_[bbox].jpg` (e.g., `0001_c1s1_001051_00.jpg`)

---

## 2. CUHK03
- **Source**: [Li et al., CVPR 2014](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- **Directory Layout**:
  ```text
  data/cuhk03/
  ├── labeled/
  │   └── [pid]/               # e.g., 0001/01.jpg
  └── detected/
      └── [pid]/               # DPM automated detections
  ```

---

## 3. MSMT17
- **Source**: [Wei et al., CVPR 2018](https://www.pkuvmc.cc/publications/msmt17.html)
- **Directory Layout**:
  ```text
  data/msmt17/
  ├── train/                   # 32,621 images
  ├── test/                    # 93,820 images
  ├── list_train.txt
  └── list_val.txt
  ```
- **File Naming Format**: `[pid]_[cam]_[frame].jpg`

---

## 4. VIPeR
- **Source**: [Gray & Tao, ECCV 2008](https://vision.soe.ucsc.edu/node/178)
- **Directory Layout**:
  ```text
  data/viper/
  ├── cam_a/                   # 632 images (000_45.bmp to 631_45.bmp)
  └── cam_b/                   # 632 images from orthogonal viewpoint
  ```
- **File Naming Format**: `[pid]_[angle].bmp`

---

## 5. GRID (Underground Station)
- **Source**: [Loy et al., CVPR 2009](https://www.qmul.ac.uk/eecs/research/eecs-research-centres/qcore/resources/datasets/)
- **Directory Layout**:
  ```text
  data/grid/
  ├── probe/                   # 250 probe images
  └── gallery/                 # 250 matched gallery + 775 distractors
  ```
- **File Naming Format**: `[pid]_[cam].bmp`

---

## 6. iLIDS-VID
- **Source**: [Wang et al., ECCV 2014](https://www.eecs.qmul.ac.uk/~jrl/iLIDS-VID.html)
- **Directory Layout**:
  ```text
  data/ilids-vid/
  ├── cam_1/
  │   └── person_001/          # Image frames
  └── cam_2/
      └── person_001/
  ```

---

## Running Multi-Dataset Evaluation
To automatically detect and evaluate all available datasets:
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_cross_dataset.py
```
Results are saved to `results/cross_dataset.json` and rendered live in the web dashboard.
