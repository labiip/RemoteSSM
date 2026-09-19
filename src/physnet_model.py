"""
PhysNet (3DCNN-based) -- Z. Yu, X. Li, G. Zhao, BMVC 2019.

结构忠实复刻作者官方 PyTorch 实现 (Zitong Yu 2019-05-05 发布, MIT License):
  3D conv 编码器 (16 -> 32 -> 64 通道, 3x3x3 conv + BN + ReLU)
  + 时间维度 encoder-decoder (两次 MaxPool3d(2,2,2) 后两次 ConvTranspose 上采样回 T)
  + 最后 channel-wise 1x1x1 conv 把特征投影为逐帧 rPPG 信号
  + AdaptiveAvgPool3d 折叠空间维度

输入: face_roi (B, T, H, W, 3), RGB float [0,1]   -- 与 RemoteSSM-v3 完全相同的输入
输出: rppg (B, T)

说明: 时间池化 300 -> 150 -> 75, 上采样 75 -> 150 -> 300, 长度恰好还原,
     AdaptiveAvgPool 输出恒等于 (B, 64, T, 1, 1), 供 1x1x1 conv 投影。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, [3, 3, 3], stride=1, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class PhysNet3DCNN(nn.Module):
    def __init__(self, frames=300, spatial_dropout=0.2, head_dropout=0.5):
        super().__init__()
        self.frames = int(frames)

        # 第一层仅在空间上卷积 (1x5x5), 避免对首帧做时间填充
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.ConvBlock2 = _conv_block(16, 32)
        self.ConvBlock3 = _conv_block(32, 64)
        self.ConvBlock4 = _conv_block(64, 64)
        self.ConvBlock5 = _conv_block(64, 64)
        self.ConvBlock6 = _conv_block(64, 64)
        self.ConvBlock7 = _conv_block(64, 64)
        self.ConvBlock8 = _conv_block(64, 64)
        self.ConvBlock9 = _conv_block(64, 64)

        # 时间解码器: 每次在时间维上采样 x2
        def _up():
            return nn.Sequential(
                nn.ConvTranspose3d(64, 64, [4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
                nn.BatchNorm3d(64),
                nn.ELU(),
            )
        self.upsample = _up()
        self.upsample2 = _up()

        # channel-wise 1x1x1 投影 -> 单通道 rPPG 波形
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.poolspa = nn.AdaptiveAvgPool3d((self.frames, 1, 1))

        self.spatial_dropout = spatial_dropout
        self.head_dropout = head_dropout

    def forward(self, face_roi):
        # face_roi: (B, T, H, W, 3) -> (B, 3, T, H, W)
        B, T = face_roi.shape[0], face_roi.shape[1]
        x = face_roi.permute(0, 4, 1, 2, 3).contiguous().float()

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(F.dropout(x, p=self.spatial_dropout, training=self.training))
        x = self.ConvBlock9(F.dropout(x, p=self.spatial_dropout, training=self.training))
        x = self.upsample(x)
        x = self.upsample2(x)

        x = self.poolspa(x)                                        # (B,64,T,1,1)
        x = self.ConvBlock10(F.dropout(x, p=self.head_dropout, training=self.training))
        return x.view(B, T)                                        # (B,T) rPPG


if __name__ == "__main__":
    m = PhysNet3DCNN(frames=300)
    n_p = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"PhysNet3DCNN params: {n_p:.2f}M")
    x = torch.randn(2, 300, 32, 32, 3)
    with torch.no_grad():
        y = m(x)
    print(f"in {tuple(x.shape)} -> out {tuple(y.shape)}  finite={bool(torch.isfinite(y).all())}")
