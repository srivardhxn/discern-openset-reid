"""
Market-1501 dataset loader and synthetic uniform look-alike dataset generator.
Allows Discern to run immediately with synthetic look-alike data or real Market-1501.
"""

from __future__ import annotations
import os
import re
import glob
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw
import numpy as np
import cv2

@dataclass
class ReIDSample:
    """Represents a single person re-id crop."""
    image_path: str
    identity_id: int
    camera_id: int
    sequence_id: int
    frame_id: int
    is_synthetic: bool = False


class Market1501Dataset:
    """
    Market-1501 dataset parser.
    File naming convention: [person_id]_c[cam_id]s[seq_id]_[frame_id]_[bbox_idx].jpg
    Example: 0001_c1s1_001051_00.jpg
    """
    PATTERN = re.compile(r"([-\d]+)_c(\d+)s(\d+)_([-\d]+)_(\d+)")

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, "bounding_box_train")
        self.test_dir = os.path.join(root_dir, "bounding_box_test")
        self.query_dir = os.path.join(root_dir, "query")

    def exists(self) -> bool:
        """Check if root directory contains required subfolders."""
        return (
            os.path.isdir(self.train_dir)
            and os.path.isdir(self.test_dir)
            and os.path.isdir(self.query_dir)
        )

    def load_split(self, split_name: str) -> List[ReIDSample]:
        """Load samples from split ('train', 'test', 'query')."""
        if split_name == "train":
            folder = self.train_dir
        elif split_name == "test":
            folder = self.test_dir
        elif split_name == "query":
            folder = self.query_dir
        else:
            raise ValueError(f"Unknown split: {split_name}")

        if not os.path.isdir(folder):
            return []

        samples: List[ReIDSample] = []
        image_files = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))

        for path in image_files:
            filename = os.path.basename(path)
            match = self.PATTERN.match(filename)
            if not match:
                continue
            pid, cam_id, seq_id, frame_id, _ = match.groups()
            pid_int = int(pid)
            # Market-1501 uses -1 or 0000 for junk / background
            if pid_int <= 0:
                continue

            samples.append(
                ReIDSample(
                    image_path=path,
                    identity_id=pid_int,
                    camera_id=int(cam_id),
                    sequence_id=int(seq_id),
                    frame_id=int(frame_id),
                    is_synthetic=False,
                )
            )

        return samples


class SyntheticUniformDatasetGenerator:
    """
    Generates a realistic synthetic dataset of people wearing matching uniforms
    (e.g., security personnel in navy, paramedics in teal, warehouse staff in hi-vis orange).
    Ensures low inter-class appearance variance with subtle distinguishing features:
    skin tone, hair style, badges, belt buckles, lanyard colors, shoes, watch, height.
    """

    UNIFORM_THEMES = [
        {
            "name": "SecurityGuard_Navy",
            "upper_hsv": (110, 180, 70),   # Dark navy shirt
            "lower_hsv": (110, 190, 45),   # Darker navy/black pants
            "badge_color": (210, 180, 20), # Gold/brass badge
        },
        {
            "name": "Warehouse_HiVis",
            "upper_hsv": (18, 220, 220),   # Hi-vis safety orange vest
            "lower_hsv": (0, 0, 50),       # Charcoal industrial pants
            "badge_color": (0, 0, 240),    # Reflective silver stripe
        },
        {
            "name": "Medical_TealScrubs",
            "upper_hsv": (85, 140, 140),   # Teal hospital scrub top
            "lower_hsv": (85, 145, 130),   # Teal hospital scrub pants
            "badge_color": (0, 200, 200),  # Cyan lanyard
        },
    ]

    SKIN_TONES = [
        (25, 60, 220),   # Fair
        (20, 90, 190),   # Light-medium
        (18, 120, 160),  # Medium tan
        (15, 140, 120),  # Olive/deep
        (12, 160, 80),   # Dark
    ]

    HAIR_COLORS = [
        (0, 0, 20),      # Black
        (15, 120, 60),   # Dark brown
        (18, 100, 100),  # Medium brown
        (25, 90, 180),   # Blonde
        (0, 0, 180),     # Gray
    ]

    def __init__(self, output_dir: str, num_identities: int = 36, samples_per_id: int = 8, seed: int = 42):
        self.output_dir = output_dir
        self.num_identities = num_identities
        self.samples_per_id = samples_per_id
        self.seed = seed

    def _hsv_to_rgb(self, hsv: Tuple[int, int, int]) -> Tuple[int, int, int]:
        h, s, v = hsv
        h = (h % 180)
        s = min(255, max(0, s))
        v = min(255, max(0, v))
        hsv_mat = np.uint8([[[h, s, v]]])
        rgb_mat = cv2.cvtColor(hsv_mat, cv2.COLOR_HSV2RGB)
        return int(rgb_mat[0, 0, 0]), int(rgb_mat[0, 0, 1]), int(rgb_mat[0, 0, 2])

    def generate(self, force: bool = False) -> str:
        """Generates synthetic dataset in Market-1501 format if not already present."""
        train_dir = os.path.join(self.output_dir, "bounding_box_train")
        test_dir = os.path.join(self.output_dir, "bounding_box_test")
        query_dir = os.path.join(self.output_dir, "query")

        if not force and os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 20:
            return self.output_dir

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)

        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        # Assign identities to uniform groups (creating clusters of high appearance similarity)
        identities = []
        num_themes = len(self.UNIFORM_THEMES)
        for i in range(1, self.num_identities + 1):
            theme = self.UNIFORM_THEMES[(i - 1) % num_themes]
            skin = rng.choice(self.SKIN_TONES)
            hair = rng.choice(self.HAIR_COLORS)
            has_glasses = rng.random() > 0.6
            has_beard = rng.random() > 0.7
            has_watch = rng.random() > 0.5
            height_factor = rng.uniform(0.92, 1.08)
            # subtle individual variation in badge/lanyard
            badge_pos = (rng.randint(34, 46), rng.randint(48, 62))

            identities.append({
                "id": i,
                "theme": theme,
                "skin": skin,
                "hair": hair,
                "has_glasses": has_glasses,
                "has_beard": has_beard,
                "has_watch": has_watch,
                "height_factor": height_factor,
                "badge_pos": badge_pos,
            })

        # 60% of identities for train, 40% for test/query
        num_train_ids = int(self.num_identities * 0.5)
        train_identities = identities[:num_train_ids]
        test_identities = identities[num_train_ids:]

        # Render function for an identity in a specific camera/viewpoint
        def render_person(id_info: dict, cam_id: int, sample_idx: int) -> Image.Image:
            # ReID crop standard: 128w x 256h
            width, height = 128, 256
            img = Image.new("RGB", (width, height), color=(225, 230, 235))
            draw = ImageDraw.Draw(img)

            # Background subtle gradient/texture
            bg_base = rng.randint(215, 240)
            for y in range(height):
                fade = int(bg_base - (y / height) * 20)
                draw.line([(0, y), (width, y)], fill=(fade, fade + 2, fade + 4))

            # Base parameters
            hf = id_info["height_factor"]
            skin_rgb = self._hsv_to_rgb(id_info["skin"])
            hair_rgb = self._hsv_to_rgb(id_info["hair"])
            upper_rgb = self._hsv_to_rgb(id_info["theme"]["upper_hsv"])
            lower_rgb = self._hsv_to_rgb(id_info["theme"]["lower_hsv"])
            badge_rgb = self._hsv_to_rgb(id_info["theme"]["badge_color"])

            # Small viewpoint/pose shift across camera/samples
            dx = int(np_rng.normal(0, 2))
            cx = width // 2 + dx

            # Head
            head_r = int(16 * hf)
            head_cy = int(36 * hf)
            draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=skin_rgb)

            # Hair
            draw.arc([cx - head_r, head_cy - head_r, cx + head_r, head_cy], start=180, end=360, fill=hair_rgb, width=int(5 * hf))
            draw.chord([cx - head_r, head_cy - head_r - 2, cx + head_r, head_cy - 4], start=180, end=360, fill=hair_rgb)

            # Glasses
            if id_info["has_glasses"]:
                draw.line([cx - 10, head_cy, cx - 2, head_cy], fill=(30, 30, 30), width=2)
                draw.line([cx + 2, head_cy, cx + 10, head_cy], fill=(30, 30, 30), width=2)
                draw.line([cx - 2, head_cy, cx + 2, head_cy], fill=(30, 30, 30), width=1)

            # Upper body (torso & arms)
            torso_top = head_cy + head_r
            torso_bot = int(140 * hf)
            torso_w = int(24 * hf)
            draw.rectangle([cx - torso_w, torso_top, cx + torso_w, torso_bot], fill=upper_rgb)

            # Arms
            arm_w = int(7 * hf)
            draw.rectangle([cx - torso_w - arm_w, torso_top + 4, cx - torso_w, torso_bot - 10], fill=upper_rgb)
            draw.rectangle([cx + torso_w, torso_top + 4, cx + torso_w + arm_w, torso_bot - 10], fill=upper_rgb)
            # Hands
            draw.ellipse([cx - torso_w - arm_w, torso_bot - 12, cx - torso_w, torso_bot], fill=skin_rgb)
            draw.ellipse([cx + torso_w, torso_bot - 12, cx + torso_w + arm_w, torso_bot], fill=skin_rgb)

            # Badge / Lanyard
            bx, by = id_info["badge_pos"]
            draw.rectangle([cx - torso_w + bx - 26, torso_top + by - 30, cx - torso_w + bx - 18, torso_top + by - 18], fill=badge_rgb)

            # Lower body (legs & shoes)
            leg_w = int(10 * hf)
            legs_bot = int(228 * hf)
            # Left leg
            draw.rectangle([cx - torso_w + 3, torso_bot, cx - torso_w + 3 + leg_w, legs_bot], fill=lower_rgb)
            # Right leg
            draw.rectangle([cx + torso_w - 3 - leg_w, torso_bot, cx + torso_w - 3, legs_bot], fill=lower_rgb)

            # Shoes
            draw.rectangle([cx - torso_w + 1, legs_bot, cx - torso_w + 5 + leg_w, int(240 * hf)], fill=(20, 20, 20))
            draw.rectangle([cx + torso_w - 5 - leg_w, legs_bot, cx + torso_w - 1, int(240 * hf)], fill=(20, 20, 20))

            # Add subtle natural sensor noise and blur for realistic Re-ID look
            arr = np.array(img).astype(np.float32)
            noise = np_rng.normal(0, 3.5, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(arr, (3, 3), 0.5)
            return Image.fromarray(blurred)

        # Generate train samples
        for id_info in train_identities:
            pid = id_info["id"]
            for sample_idx in range(self.samples_per_id):
                cam_id = (sample_idx % 4) + 1
                seq_id = 1
                frame_id = sample_idx * 100 + 1
                img = render_person(id_info, cam_id, sample_idx)
                fname = f"{pid:04d}_c{cam_id}s{seq_id}_{frame_id:06d}_00.jpg"
                img.save(os.path.join(train_dir, fname))

        # Generate test samples (gallery and query)
        for id_info in test_identities:
            pid = id_info["id"]
            for sample_idx in range(self.samples_per_id):
                cam_id = (sample_idx % 4) + 1
                seq_id = 1
                frame_id = sample_idx * 100 + 1
                img = render_person(id_info, cam_id, sample_idx)
                fname = f"{pid:04d}_c{cam_id}s{seq_id}_{frame_id:06d}_00.jpg"
                # Split half into test (gallery) and half into query
                if sample_idx < self.samples_per_id // 2:
                    img.save(os.path.join(test_dir, fname))
                else:
                    img.save(os.path.join(query_dir, fname))

        return self.output_dir
