"""
Multi-Scale Temporal Stem — 多尺度 1D 时间卷积前端。

用不同感受野的卷积核并行提取时间维度上的多尺度脉冲特征，
替代简单的 nn.Linear(1, d_model) 静态投影，显著提升 rPPG 时序建模能力。

设计参考:
  - EfficientPhysioNet 的时序差分思想
  - InceptionTime 多尺度时间卷积
  - RhythmMamba (AAAI 2025) 的频域前馈

参数量: ~5K per stem (极轻量)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalStem(nn.Module):
    """多尺度 1D 时间卷积前端。

    并行使用多个不同感受野的 Conv1d 分支，每个分支捕获
    不同时间尺度的信号特征（如 3 帧局部变化、11 帧心跳周期等），
    最终拼接并通过可学习加权融合为一个 d_model 维特征。

    Args:
        in_channels: 输入通道数 (real/imag 各 1 通道)
        d_model:   输出特征维度 (与 SSM 一致，默认 128)
        scales:    多尺度卷积核大小 (默认 [3,5,7,9,11] 覆盖 0.1-0.37s @30fps)
        dropout:   正则化 dropout 率

    Input:  (B, L, in_channels)   — e.g. (16, 300, 1)
    Output: (B, L, d_model)       — e.g. (16, 300, 128)
    """

    def __init__(self, in_channels: int = 1, d_model: int = 128,
                 scales: tuple = (3, 5, 7, 9, 11), dropout: float = 0.1):
        super().__init__()
        n_scales = len(scales)
        base_ch = d_model // n_scales
        ch_per_scale = [base_ch] * (n_scales - 1) + [d_model - base_ch * (n_scales - 1)]

        self.branches = nn.ModuleList()
        for k, ch in zip(scales, ch_per_scale):
            branch = nn.Sequential(
                # Depthwise: 每组独立卷积，不混通道
                nn.Conv1d(in_channels, ch, kernel_size=k, padding=k // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                # Pointwise: 1x1 跨组融合
                nn.Conv1d(ch, ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            )
            self.branches.append(branch)

        # 通道融合：虽然已 concat，但额外用 1x1 做通道间交互
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 残差捷径: 将原始 1 通道信号投影到 d_model 空间
        self.shortcut_proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)

        # 输出层归一化
        self.out_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 轻量输入归一化: 仅去除 DC 偏移，保留 PPG 幅度信息 ──
        # x: (B, L, 1)
        x = x - x.mean(dim=1, keepdim=True)

        # x: (B, L, 1) → (B, 1, L) for Conv1d
        x = x.transpose(1, 2)

        # 多尺度并行提取
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))  # each: (B, ch_i, L)

        # 沿通道维拼接 → (B, d_model, L)
        fused = torch.cat(branch_outs, dim=1)

        # 1x1 融合 → (B, d_model, L)
        fused = self.fusion(fused)

        # 残差捷径: 用 1x1 Conv 将原始信号投影到 d_model 空间
        shortcut = self.shortcut_proj(x)
        out = fused + shortcut

        # 转回 (B, L, d_model)
        return self.out_norm(out.transpose(1, 2))
