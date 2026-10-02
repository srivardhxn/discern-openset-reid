"""
Reusable feature extraction module and latency/parameter benchmarking for Discern.
Supports single image (file path, PIL Image, cv2 ndarray) or batch tensor input.
Returns L2-normalized 512-dimensional embeddings.
"""

from __future__ import annotations
import os
import time
from typing import List, Union, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from .osnet import OSNetReID

# Standard Re-ID evaluation image preprocessing (256 height x 128 width)
REID_TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FeatureExtractor:
    """
    Standardized inference wrapper for OSNet ReID embedding extraction.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_stripes: bool = True,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_stripes = use_stripes
        self.model = OSNetReID(use_stripes=use_stripes).to(self.device)

        if weights_path and os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            # Filter classifier keys if present
            filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
            self.model.load_state_dict(filtered_dict, strict=False)

        self.model.eval()
        self.transform = REID_TRANSFORM

    def _prepare_image(self, img_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Converts image path, cv2 numpy array, or PIL image into normalized tensor."""
        if isinstance(img_input, str):
            if not os.path.isfile(img_input):
                raise FileNotFoundError(f"Image not found at: {img_input}")
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            # If BGR from cv2, convert to RGB
            if len(img_input.shape) == 3 and img_input.shape[2] == 3:
                rgb_arr = img_input[:, :, ::-1]
                pil_img = Image.fromarray(rgb_arr)
            else:
                pil_img = Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")

        tensor = self.transform(pil_img) # (3, 256, 128)
        return tensor

    @torch.no_grad()
    def extract(self, img_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extracts 512-d L2-normalized embedding for a single image.
        Returns: 1D numpy array of shape (512,).
        """
        tensor = self._prepare_image(img_input).unsqueeze(0).to(self.device)
        emb = self.model.extract_features(tensor)
        return emb.cpu().numpy()[0]

    @torch.no_grad()
    def extract_batch(self, img_inputs: List[Union[str, Image.Image, np.ndarray]], batch_size: int = 32) -> np.ndarray:
        """
        Extracts embeddings for a list of images in mini-batches.
        Returns: 2D numpy array of shape (N, 512).
        """
        all_embeddings = []
        for i in range(0, len(img_inputs), batch_size):
            batch_slice = img_inputs[i:i + batch_size]
            tensors = [self._prepare_image(item) for item in batch_slice]
            batch_tensor = torch.stack(tensors).to(self.device)
            emb = self.model.extract_features(batch_tensor)
            all_embeddings.append(emb.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32)

        return np.concatenate(all_embeddings, axis=0)


def benchmark_model(
    model: OSNetReID,
    device_name: str = "cpu",
    warmup_iters: int = 10,
    test_iters: int = 50,
) -> Dict[str, Any]:
    """
    Benchmarks parameter count and per-image inference latency.
    """
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(1, 3, 256, 128, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model.extract_features(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Latency timing
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(test_iters):
            _ = model.extract_features(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / test_iters) * 1000.0

    return {
        "device": device_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameters_million": round(total_params / 1e6, 3),
        "latency_ms_per_image": round(avg_latency_ms, 2),
        "throughput_fps": round(1000.0 / avg_latency_ms, 1),
    }
