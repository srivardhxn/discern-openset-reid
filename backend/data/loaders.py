"""
Unified Dataset Loaders for Person Re-Identification Benchmarks:
- Market-1501
- CUHK03
- MSMT17
- VIPeR
- GRID
- iLIDS-VID
- Synthetic Uniform Look-Alikes

Detects existing datasets in data/, parses annotation conventions,
and provides standardized ReIDSample lists. Gracefully flags missing datasets.
"""

from __future__ import annotations
import os
import re
import glob
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from .dataset import ReIDSample, Market1501Dataset, SyntheticUniformDatasetGenerator


@dataclass
class DatasetMetadata:
    name: str
    display_name: str
    path: str
    exists: bool
    num_samples: int = 0
    num_identities: int = 0
    num_cameras: int = 0
    download_url: str = ""
    description: str = ""


class VIPeRDataset:
    """
    VIPeR dataset parser.
    Directory structure:
      data/viper/
        cam_a/ (e.g. 000_45.bmp)
        cam_b/ (e.g. 000_90.bmp)
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.cam_a_dir = os.path.join(root_dir, "cam_a")
        self.cam_b_dir = os.path.join(root_dir, "cam_b")

    def exists(self) -> bool:
        return os.path.isdir(self.cam_a_dir) and os.path.isdir(self.cam_b_dir)

    def load_samples(self) -> List[ReIDSample]:
        if not self.exists():
            return []
        samples = []
        for cam_idx, cam_dir in enumerate([self.cam_a_dir, self.cam_b_dir], start=1):
            for p in sorted(glob.glob(os.path.join(cam_dir, "*.*"))):
                fname = os.path.basename(p)
                # Convention: [id]_[viewpoint].bmp (e.g. 001_45.bmp)
                match = re.match(r"^(\d+)_", fname)
                if match:
                    pid = int(match.group(1))
                    samples.append(ReIDSample(
                        image_path=p,
                        identity_id=pid,
                        camera_id=cam_idx,
                        sequence_id=1,
                        frame_id=1
                    ))
        return samples


class CUHK03Dataset:
    """
    CUHK03 dataset parser.
    Directory structure:
      data/cuhk03/
        labeled/ or detected/
          [pid]/ (e.g. 0001/)
            *.jpg
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def exists(self) -> bool:
        if not os.path.isdir(self.root_dir):
            return False
        # Check for subfolders or images
        return len(glob.glob(os.path.join(self.root_dir, "**", "*.jpg"), recursive=True)) > 50

    def load_samples(self) -> List[ReIDSample]:
        if not self.exists():
            return []
        samples = []
        image_files = sorted(glob.glob(os.path.join(self.root_dir, "**", "*.jpg"), recursive=True))
        for p in image_files:
            fname = os.path.basename(p)
            # Convention: [pid]_[cam]_[seq]_[frame].jpg or [pid]/[img].jpg
            parent_dir = os.path.basename(os.path.dirname(p))
            if parent_dir.isdigit():
                pid = int(parent_dir)
                samples.append(ReIDSample(image_path=p, identity_id=pid, camera_id=1, sequence_id=1, frame_id=1))
            else:
                match = re.match(r"^(\d+)_", fname)
                if match:
                    pid = int(match.group(1))
                    samples.append(ReIDSample(image_path=p, identity_id=pid, camera_id=1, sequence_id=1, frame_id=1))
        return samples


class MSMT17Dataset:
    """
    MSMT17 dataset parser.
    Directory structure:
      data/msmt17/
        train/ (e.g. 0001_001_01.jpg)
        test/
        list_train.txt, list_val.txt
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, "train")
        self.test_dir = os.path.join(root_dir, "test")

    def exists(self) -> bool:
        return os.path.isdir(self.train_dir) or os.path.isdir(self.test_dir)

    def load_samples(self) -> List[ReIDSample]:
        if not self.exists():
            return []
        samples = []
        for target_dir in [self.train_dir, self.test_dir]:
            if not os.path.isdir(target_dir):
                continue
            for p in sorted(glob.glob(os.path.join(target_dir, "*.jpg"))):
                fname = os.path.basename(p)
                # Convention: [pid]_[cam]_[frame].jpg
                match = re.match(r"^(\d+)_(\d+)_", fname)
                if match:
                    pid, cam = int(match.group(1)), int(match.group(2))
                    samples.append(ReIDSample(image_path=p, identity_id=pid, camera_id=cam, sequence_id=1, frame_id=1))
        return samples


class GRIDDataset:
    """
    GRID (Underground Station) dataset parser.
    Directory structure:
      data/grid/
        probe/ (e.g. 0001_1.bmp)
        gallery/ (e.g. 0001_2.bmp)
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.probe_dir = os.path.join(root_dir, "probe")
        self.gallery_dir = os.path.join(root_dir, "gallery")

    def exists(self) -> bool:
        return os.path.isdir(self.probe_dir) and os.path.isdir(self.gallery_dir)

    def load_samples(self) -> List[ReIDSample]:
        if not self.exists():
            return []
        samples = []
        for cam_id, folder in enumerate([self.probe_dir, self.gallery_dir], start=1):
            for p in sorted(glob.glob(os.path.join(folder, "*.*"))):
                fname = os.path.basename(p)
                match = re.match(r"^(\d+)_", fname)
                if match:
                    pid = int(match.group(1))
                    samples.append(ReIDSample(image_path=p, identity_id=pid, camera_id=cam_id, sequence_id=1, frame_id=1))
        return samples


class ILIDSVIDDataset:
    """
    iLIDS-VID video person re-identification crop parser.
    Directory structure:
      data/ilids-vid/
        cam_1/
          person_001/
            *.png
        cam_2/
          person_001/
            *.png
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.cam1_dir = os.path.join(root_dir, "cam_1")
        self.cam2_dir = os.path.join(root_dir, "cam_2")

    def exists(self) -> bool:
        return os.path.isdir(self.cam1_dir) and os.path.isdir(self.cam2_dir)

    def load_samples(self) -> List[ReIDSample]:
        if not self.exists():
            return []
        samples = []
        for cam_id, cam_folder in enumerate([self.cam1_dir, self.cam2_dir], start=1):
            for p_folder in sorted(glob.glob(os.path.join(cam_folder, "person_*"))):
                folder_name = os.path.basename(p_folder)
                match = re.search(r"(\d+)", folder_name)
                if match:
                    pid = int(match.group(1))
                    # Sample up to 5 frames per person to represent the sequence
                    frames = sorted(glob.glob(os.path.join(p_folder, "*.png")) + glob.glob(os.path.join(p_folder, "*.jpg")))[:5]
                    for f_idx, f_path in enumerate(frames):
                        samples.append(ReIDSample(image_path=f_path, identity_id=pid, camera_id=cam_id, sequence_id=1, frame_id=f_idx))
        return samples


DATASET_REGISTRY = [
    {
        "id": "market1501",
        "display_name": "Market-1501",
        "default_dir": "data/sample_market1501",
        "alt_dir": "data/Market-1501-v15.09.15",
        "cls": Market1501Dataset,
        "download_url": "https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html",
        "description": "Supermarket camera network benchmark (1,501 identities, 6 cameras). Primary source.",
    },
    {
        "id": "cuhk03",
        "display_name": "CUHK03",
        "default_dir": "data/cuhk03",
        "alt_dir": "data/cuhk03_release",
        "cls": CUHK03Dataset,
        "download_url": "https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html",
        "description": "Campus surveillance benchmark (1,467 identities, 10 camera pairs with manual and DPM bboxes).",
    },
    {
        "id": "msmt17",
        "display_name": "MSMT17",
        "default_dir": "data/msmt17",
        "alt_dir": "data/MSMT17_V1",
        "cls": MSMT17Dataset,
        "download_url": "https://www.pkuvmc.cc/publications/msmt17.html",
        "description": "Large-scale multi-scene multi-time benchmark (4,101 identities, 15 cameras, complex lighting).",
    },
    {
        "id": "viper",
        "display_name": "VIPeR",
        "default_dir": "data/viper",
        "alt_dir": "data/VIPeR",
        "cls": VIPeRDataset,
        "download_url": "https://vision.soe.ucsc.edu/node/178",
        "description": "Classic benchmark (632 pedestrian identity pairs across two viewpoints with severe viewpoint shift).",
    },
    {
        "id": "grid",
        "display_name": "GRID",
        "default_dir": "data/grid",
        "alt_dir": "data/underground_reid",
        "cls": GRIDDataset,
        "download_url": "https://www.qmul.ac.uk/eecs/research/eecs-research-centres/qcore/resources/datasets/",
        "description": "Underground station benchmark (250 probe-gallery pairs + 775 distractor impostors).",
    },
    {
        "id": "ilids_vid",
        "display_name": "iLIDS-VID",
        "default_dir": "data/ilids-vid",
        "alt_dir": "data/iLIDS-VID",
        "cls": ILIDSVIDDataset,
        "download_url": "https://www.eecs.qmul.ac.uk/~jrl/iLIDS-VID.html",
        "description": "Airport arrival hall CCTV benchmark (300 identities under severe occlusion).",
    },
]


def discover_available_datasets(base_project_dir: str) -> List[Dict[str, Any]]:
    """Discovers which datasets exist locally in data/ and reports metadata."""
    results = []
    for entry in DATASET_REGISTRY:
        p1 = os.path.join(base_project_dir, entry["default_dir"])
        p2 = os.path.join(base_project_dir, entry["alt_dir"])
        resolved_path = p1 if os.path.isdir(p1) else (p2 if os.path.isdir(p2) else p1)

        loader = entry["cls"](resolved_path)
        exists = loader.exists()
        num_samples = 0
        num_ids = 0

        if exists:
            if hasattr(loader, "load_split"):
                # Market1501
                samples = loader.load_split("train") + loader.load_split("test") + loader.load_split("query")
            else:
                samples = loader.load_samples()
            num_samples = len(samples)
            num_ids = len(set(s.identity_id for s in samples))

        results.append({
            "id": entry["id"],
            "display_name": entry["display_name"],
            "path": os.path.relpath(resolved_path, base_project_dir),
            "exists": exists,
            "num_samples": num_samples,
            "num_identities": num_ids,
            "download_url": entry["download_url"],
            "description": entry["description"],
        })
    return results
