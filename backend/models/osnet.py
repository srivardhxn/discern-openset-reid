"""
Omni-Scale Network (OSNet) backbone with Horizontal Stripe Feature Pooling.
Specifically designed for person re-identification under low appearance variance.
Outputs an L2-normalized 512-dimensional embedding consisting of:
- Global descriptor (128-d)
- Stripe 1 / Top: Head, neck, upper chest (128-d)
- Stripe 2 / Mid: Torso, pocket badges, waistline (128-d)
- Stripe 3 / Bot: Legs, hems, footwear (128-d)
Total: 512 dimensions.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Standard Conv-BN-ReLU block."""
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 conv using depthwise + pointwise convolution."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class OSBlock(nn.Module):
    """
    Omni-Scale Feature Learning Block.
    Captures multi-scale features via streams of receptive field depths (scales 1, 2, 3, 4)
    and fuses them with dynamic channel-wise gate.
    """
    def __init__(self, in_c: int, out_c: int, bottleneck_reduction: int = 4):
        super().__init__()
        mid_c = out_c // bottleneck_reduction
        self.conv1 = ConvBlock(in_c, mid_c, kernel_size=1, stride=1, padding=0)

        # Scale 1: 1 conv (RF = 3x3)
        self.scale1 = LightConv3x3(mid_c, mid_c)
        # Scale 2: 2 convs (RF = 5x5)
        self.scale2 = nn.Sequential(
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c)
        )
        # Scale 3: 3 convs (RF = 7x7)
        self.scale3 = nn.Sequential(
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c)
        )
        # Scale 4: 4 convs (RF = 9x9)
        self.scale4 = nn.Sequential(
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c)
        )

        # Aggregation Gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, mid_c, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv2 = ConvBlock(mid_c, out_c, kernel_size=1, stride=1, padding=0)

        # Residual shortcut
        if in_c != out_c:
            self.shortcut = ConvBlock(in_c, out_c, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        feat = self.conv1(x)

        s1 = self.scale1(feat)
        s2 = self.scale2(feat)
        s3 = self.scale3(feat)
        s4 = self.scale4(feat)

        # Multi-scale fusion with attention gating
        s_sum = s1 + s2 + s3 + s4
        attn = self.gate(s_sum)
        fused = s1 * attn + s2 * (1.0 - attn) + s3 * attn + s4 * (1.0 - attn)

        out = self.conv2(fused)
        return F.relu(out + res)


class OSNetReID(nn.Module):
    """
    Compact OSNet with Horizontal Stripe Feature Extraction.
    Input size: (B, 3, 256, 128)
    Feature map before head: (B, 256, 16, 8)
    Output: 512-dim L2-normalized embedding.
    """
    def __init__(self, num_classes: int = 0, use_stripes: bool = True, feature_dim: int = 512):
        super().__init__()
        self.use_stripes = use_stripes
        self.feature_dim = feature_dim

        # Initial stem
        self.stem = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64x32
        )

        # Stage 1 (64x32 -> 64x32)
        self.stage1 = nn.Sequential(
            OSBlock(32, 64),
            OSBlock(64, 64)
        )
        self.down1 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1)  # 32x16

        # Stage 2 (32x16 -> 32x16)
        self.stage2 = nn.Sequential(
            OSBlock(64, 128),
            OSBlock(128, 128)
        )
        self.down2 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1) # 16x8

        # Stage 3 (16x8 -> 16x8)
        self.stage3 = nn.Sequential(
            OSBlock(128, 256),
            OSBlock(256, 256)
        )

        # Heads: Global and 3 Horizontal Stripes
        # When use_stripes is True:
        # global: 128, stripe1: 128, stripe2: 128, stripe3: 128 => 512 total
        # When use_stripes is False (ablation):
        # global projected directly to 512
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_stripes:
            self.global_proj = nn.Sequential(
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128)
            )
            self.stripe1_proj = nn.Sequential(
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128)
            )
            self.stripe2_proj = nn.Sequential(
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128)
            )
            self.stripe3_proj = nn.Sequential(
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128)
            )
        else:
            self.global_proj = nn.Sequential(
                nn.Linear(256, 512, bias=False),
                nn.BatchNorm1d(512)
            )

        # Classifier for training classification/margin loss
        if num_classes > 0:
            self.classifier = nn.Linear(512, num_classes, bias=False)
        else:
            self.classifier = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-d L2-normalized embedding."""
        feat_map = self.stem(x)
        feat_map = self.stage1(feat_map)
        feat_map = self.down1(feat_map)
        feat_map = self.stage2(feat_map)
        feat_map = self.down2(feat_map)
        feat_map = self.stage3(feat_map) # (B, 256, H=16, W=8)

        B, C, H, W = feat_map.shape

        if not self.use_stripes:
            glob = self.global_pool(feat_map).view(B, C)
            emb = self.global_proj(glob)
            return F.normalize(emb, p=2, dim=-1)

        # Global feature
        glob = self.global_pool(feat_map).view(B, C)
        g_emb = self.global_proj(glob)

        # 3 horizontal stripes along height H:
        # stripe 1: top 1/3 (0:H//3) -> head/neck/upper collar
        # stripe 2: middle 1/3 (H//3:2*H//3) -> torso/chest badge/belt
        # stripe 3: bottom 1/3 (2*H//3:) -> pants/shoes
        h1 = H // 3
        h2 = (2 * H) // 3
        s1 = F.adaptive_avg_pool2d(feat_map[:, :, 0:h1, :], (1, 1)).view(B, C)
        s2 = F.adaptive_avg_pool2d(feat_map[:, :, h1:h2, :], (1, 1)).view(B, C)
        s3 = F.adaptive_avg_pool2d(feat_map[:, :, h2:, :], (1, 1)).view(B, C)

        s1_emb = self.stripe1_proj(s1)
        s2_emb = self.stripe2_proj(s2)
        s3_emb = self.stripe3_proj(s3)

        # Concat: 128 + 128 + 128 + 128 = 512
        full_emb = torch.cat([g_emb, s1_emb, s2_emb, s3_emb], dim=-1)
        return F.normalize(full_emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.extract_features(x)
        logits = None
        if self.classifier is not None:
            # Cosine-based classification for ArcFace / Linear
            w = F.normalize(self.classifier.weight, p=2, dim=-1)
            logits = F.linear(emb, w)
        return emb, logits
