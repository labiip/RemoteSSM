"""Frame Stem: 轻量 2D CNN 逐帧空间特征提取器。

参考 RhythmMamba (AAAI 2025) 的 Frame Stem 设计:
将人脸 ROI 图像序列逐帧送入浅层 2D CNN，
提取肤色变化、微动作等空间特征，
输出 (B, T, d_model) 特征序列供 SSM 时序建模。

参数: ~16K, 极轻量级。
"""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """残差块: Conv→BN→GELU→Conv→BN + skip → GELU (stride=1 only)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class FrameStem(nn.Module):
    """逐帧 2D CNN: (B, T, H, W, 3) → (B, T, d_model)

    六层 Conv2d + 两个残差块，渐进下采样 32→16→8→4:
      - Block 1: 5×5 stride-2  32→16 (宽感受野捕获皮肤纹理)
      - Block 2: 两个 3×3 stride-1 残差块 @ 16×16
      - Block 3: 3×3 stride-2  16→8
      - Block 4: 两个 3×3 stride-1 残差块 @ 8×8
      - Block 5: 3×3 stride-2  8→4
      - AdaptiveAvgPool → FC → d_model
    """
    def __init__(self, d_model=128, roi_size=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.roi_size = roi_size

        # Block 1: 32×32×3 → 16×16×24
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        # Block 2: 16×16×24 → 16×16×24 (residual ×2)
        self.res1 = nn.Sequential(
            _ResBlock(24),
            nn.Dropout2d(dropout),
            _ResBlock(24),
        )
        # Block 3: 16×16×24 → 8×8×48
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        # Block 4: 8×8×48 → 8×8×48 (residual ×2)
        self.res2 = nn.Sequential(
            _ResBlock(48),
            nn.Dropout2d(dropout),
            _ResBlock(48),
        )
        # Block 5: 8×8×48 → 4×4×96
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)     # → 1×1×96
        self.fc = nn.Linear(96, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, H, W, 3) float [0,1]
        B, T = x.shape[:2]
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, 3, self.roi_size, self.roi_size)

        x = self.conv1(x)        # (BT, 24, 16, 16)
        x = self.res1(x)         # (BT, 24, 16, 16)
        x = self.conv2(x)        # (BT, 48, 8, 8)
        x = self.res2(x)         # (BT, 48, 8, 8)
        x = self.conv3(x)        # (BT, 96, 4, 4)
        x = self.pool(x)         # (BT, 96, 1, 1)
        x = x.view(B * T, -1)    # (BT, 96)
        x = self.dropout(x)
        x = self.fc(x)           # (BT, d_model)
        x = x.view(B, T, -1)     # (B, T, d_model)
        return x
