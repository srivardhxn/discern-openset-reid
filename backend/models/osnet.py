"""
Omni-Scale Network (OSNet x0.5) Backbone with Horizontal Stripe Feature Pooling.
Reference:
- Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV 2019.
- Zhou et al. Learning Generalisable Omni-Scale Representations for Person Re-Identification. TPAMI 2021.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (linear, no ReLU)."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 conv using 1x1 pointwise + 3x3 depthwise convolution."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""
    def __init__(self, in_channels: int, num_gates: Optional[int] = None, reduction: int = 16):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.global_avgpool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.gate_activation(w)
        return x * w


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    def __init__(self, in_channels: int, out_channels: int, bottleneck_reduction: int = 4):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSNetReID(nn.Module):
    """
    Omni-Scale Network (OSNet x1.0 and x0.5) for Open-Set Person Re-Identification.
    Supports:
    - Proven Bag of Tricks (BoT) BNNeck head for strong Re-ID metric learning
    - Dual backbone width multipliers: x1.0 (standard 2.2M) and x0.5 (compact 0.6M)
    - Horizontal stripe pooling heads (3 vertical partitions: head, torso, legs)
    - Auto-detection of backbone scale and head config from pretrained checkpoint
    """
    def __init__(
        self,
        num_classes: int = 0,
        width_mult: float = 1.0,
        use_stripes: bool = False,
        use_bnneck: bool = True,
        pretrained_path: Optional[str] = None,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_stripes = use_stripes
        self.use_bnneck = use_bnneck
        self.num_classes = num_classes

        # Auto-detect width_mult and head settings if a checkpoint is provided
        if pretrained_path and os.path.isfile(pretrained_path):
            try:
                ckpt = torch.load(pretrained_path, map_location="cpu")
                # Look for conv1.conv.weight in ckpt
                for k, v in ckpt.items():
                    clean_k = k[7:] if k.startswith("module.") else k
                    if clean_k == "conv1.conv.weight":
                        if v.shape[0] == 32:
                            width_mult = 0.5
                        elif v.shape[0] == 64:
                            width_mult = 1.0
                        elif v.shape[0] == 48:
                            width_mult = 0.75
                        break
                    elif clean_k == "stripe1_proj.0.weight" and not use_stripes:
                        use_stripes = True
            except Exception:
                pass

        self.width_mult = width_mult
        self.use_stripes = use_stripes

        # Backbone channels based on width multiplier
        if abs(width_mult - 1.0) < 1e-3:
            channels = [64, 256, 384, 512]
        elif abs(width_mult - 0.75) < 1e-3:
            channels = [48, 192, 288, 384]
        elif abs(width_mult - 0.5) < 1e-3:
            channels = [32, 128, 192, 256]
        elif abs(width_mult - 0.25) < 1e-3:
            channels = [16, 64, 96, 128]
        else:
            channels = [int(64 * width_mult), int(256 * width_mult), int(384 * width_mult), int(512 * width_mult)]

        self.channels = channels

        # Convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(channels[0], channels[1], num_blocks=2, reduce_spatial_size=True)
        self.conv3 = self._make_layer(channels[1], channels[2], num_blocks=2, reduce_spatial_size=True)
        self.conv4 = self._make_layer(channels[2], channels[3], num_blocks=2, reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Official fully-connected layer (Linear channels[3]->512 + BN + ReLU)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
        )

        # Optional Horizontal Stripe Features (head, torso, legs)
        if self.use_stripes:
            self.stripe1_proj = nn.Sequential(
                nn.Linear(channels[3], 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )
            self.stripe2_proj = nn.Sequential(
                nn.Linear(channels[3], 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )
            self.stripe3_proj = nn.Sequential(
                nn.Linear(channels[3], 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )
            self.stripe_fusion = nn.Sequential(
                nn.Linear(self.feature_dim + 384, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
            )

        # Bag of Tricks BNNeck layer
        if self.use_bnneck:
            self.bnneck = nn.BatchNorm1d(self.feature_dim)
            self.bnneck.bias.requires_grad_(False)
            nn.init.constant_(self.bnneck.weight, 1.0)
            nn.init.constant_(self.bnneck.bias, 0.0)

        # Classifier head for identity classification during training
        if num_classes > 0:
            self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = None

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, reduce_spatial_size: bool) -> nn.Sequential:
        layers = []
        layers.append(OSBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(OSBlock(out_channels, out_channels))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )
        return nn.Sequential(*layers)

    def load_pretrained(self, weights_path: str):
        """Loads official OSNet weights into backbone layers with full parameter coverage."""
        if not os.path.isfile(weights_path):
            print(f"[WARN] Pretrained weights file not found: {weights_path}")
            return
        sd = torch.load(weights_path, map_location="cpu")
        model_dict = self.state_dict()
        filtered = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("classifier."):
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered[k] = v
        model_dict.update(filtered)
        self.load_state_dict(model_dict, strict=False)
        print(f"[OK] Successfully loaded {len(filtered)} pretrained parameters into OSNetReID ({weights_path}).")

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature map of shape (B, channels[3], 16, 8)."""
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Computes L2-normalized 512-dimensional embedding."""
        fm = self.featuremaps(x)
        v = self.global_avgpool(fm).flatten(1)
        global_feat = self.fc(v)

        if not self.use_stripes:
            feat = global_feat
        else:
            # 3 vertical spatial partitions: 0:5 (head), 5:11 (torso), 11:16 (legs)
            s1 = F.adaptive_avg_pool2d(fm[:, :, 0:5, :], 1).flatten(1)
            s2 = F.adaptive_avg_pool2d(fm[:, :, 5:11, :], 1).flatten(1)
            s3 = F.adaptive_avg_pool2d(fm[:, :, 11:16, :], 1).flatten(1)

            s1_feat = self.stripe1_proj(s1)
            s2_feat = self.stripe2_proj(s2)
            s3_feat = self.stripe3_proj(s3)

            stripe_cat = torch.cat([global_feat, s1_feat, s2_feat, s3_feat], dim=1)
            fused = self.stripe_fusion(stripe_cat)
            feat = global_feat + 0.3 * fused

        # If BNNeck is present, evaluate with post-BN feature (BoT standard)
        if hasattr(self, "bnneck") and self.use_bnneck:
            feat = self.bnneck(feat)

        return F.normalize(feat, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass:
        - Training: returns (raw_feature_for_triplet, logits_for_ce)
        - Evaluation: returns (normalized_feature, None)
        """
        fm = self.featuremaps(x)
        v = self.global_avgpool(fm).flatten(1)
        global_feat = self.fc(v)

        if self.use_stripes:
            s1 = F.adaptive_avg_pool2d(fm[:, :, 0:5, :], 1).flatten(1)
            s2 = F.adaptive_avg_pool2d(fm[:, :, 5:11, :], 1).flatten(1)
            s3 = F.adaptive_avg_pool2d(fm[:, :, 11:16, :], 1).flatten(1)
            s1_feat = self.stripe1_proj(s1)
            s2_feat = self.stripe2_proj(s2)
            s3_feat = self.stripe3_proj(s3)
            stripe_cat = torch.cat([global_feat, s1_feat, s2_feat, s3_feat], dim=1)
            fused = self.stripe_fusion(stripe_cat)
            feat = global_feat + 0.3 * fused
        else:
            feat = global_feat

        if not self.training:
            if hasattr(self, "bnneck") and self.use_bnneck:
                feat = self.bnneck(feat)
            return F.normalize(feat, p=2, dim=1), None

        # In training mode with BNNeck
        if hasattr(self, "bnneck") and self.use_bnneck:
            feat_bn = self.bnneck(feat)
        else:
            feat_bn = feat

        if self.classifier is not None:
            logits = self.classifier(feat_bn)
        else:
            logits = None

        return feat, logits
