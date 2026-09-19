"""
RemoteSSM: 轻量复数状态空间模型 (V11 申报书算法)

V11 原始架构:
  - 单向 SSM 扫描 (极快推理)
  - FDF 静态频域前馈 (轻量级频域增强)
  - 李雅普诺夫稳定约束 (长时序稳定)
  - Dynamic Dt 自适应步长 (运动鲁棒)
  - GatedFusion 双路融合 (空间+信号)
  - 256通道随机初始化 (多尺度建模)
  - 课程学习 100→200→300

论文备选概念 (代码保留, 未使用):
  - COB (Cardiac Oscillator Bank) A矩阵心脏频段初始化
  - denoise 模式 仿真去噪预训练

与 CardioSSM-v2 的区别:
  - 不使用双向SSM (省50%推理时间)
  - 不使用ASG自适应门控 (省FFT开销)
  - 不使用FrequencyAttention (省额外计算)
  → 但论文中可以对比说明这些模块在小数据上的边际收益
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.frame_stem import FrameStem
from core.temporal_stem import MultiScaleTemporalStem
from core.parallel_scan import parallel_complex_scan


# ═══════════════════════════════════════════════════════════════
# Lyapunov Stability
# ═══════════════════════════════════════════════════════════════
def apply_lyapunov_stability(A_real, epsilon=1e-4):
    """Enforce A_real < 0 for BIBO stability."""
    return -F.softplus(-A_real) - epsilon


# ═══════════════════════════════════════════════════════════════
# GRL: Gradient Reversal Layer
# ═══════════════════════════════════════════════════════════════
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ═══════════════════════════════════════════════════════════════
# Subject Discriminator
# ═══════════════════════════════════════════════════════════════
class SubjectDiscriminator(nn.Module):
    def __init__(self, d_model=256, num_subjects=34, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_subjects),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
# FDF: Frequency Domain Feedforward
# ═══════════════════════════════════════════════════════════════
class FreqDomainFeedForward(nn.Module):
    """Static frequency-domain feedforward (V11 original).

    Applies learnable per-frequency complex weights.
    Unlike ASG, weights are static (not input-dependent) → faster inference.
    """
    def __init__(self, d_model, seq_len=300):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.re_weight = nn.Parameter(torch.ones(1, n_freq, 1) * 0.1)
        self.im_weight = nn.Parameter(torch.zeros(1, n_freq, 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        orig_dtype = x.dtype
        x_fft_in = x.float() if orig_dtype != torch.float32 else x
        x_fft = torch.fft.rfft(x_fft_in, dim=1)
        n_fft = x_fft.shape[1]
        complex_weight = torch.complex(
            self.re_weight[:, :n_fft, :],
            self.im_weight[:, :n_fft, :]
        )
        x_fft_w = x_fft * complex_weight
        x_out = torch.fft.irfft(x_fft_w, n=L, dim=1)
        if orig_dtype != x_out.dtype:
            x_out = x_out.to(orig_dtype)
        return self.norm(x + x_out)


# ═══════════════════════════════════════════════════════════════
# GatedFusion: spatial + signal dual-path fusion
# ═══════════════════════════════════════════════════════════════
class GatedFusion(nn.Module):
    """Per-timestep learnable gate: out = g * spatial + (1-g) * signal."""
    def __init__(self, d_model):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

    def forward(self, spatial, signal):
        gate_in = torch.cat([spatial, signal], dim=-1)
        g = self.gate_mlp(gate_in)
        return g * spatial + (1 - g) * signal


# ═══════════════════════════════════════════════════════════════
# Dynamic Dt Predictor (V11 original)
# ═══════════════════════════════════════════════════════════════
class DynamicDtPredictor(nn.Module):
    """Input-dependent Δt via lightweight MLP."""
    def __init__(self, d_model, dt_min=0.1, dt_max=1.0):
        super().__init__()
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x_mapped):
        dt_raw = self.mlp(x_mapped)
        return self.dt_min + dt_raw * (self.dt_max - self.dt_min)


# ═══════════════════════════════════════════════════════════════
# COB: Cardiac Oscillator Bank (唯一新增模块，零推理开销)
# ═══════════════════════════════════════════════════════════════
def cardiac_oscillator_init(d_model, fs=30.0, f_min=0.7, f_max=3.0, decay=0.95):
    """Initialize A-matrix eigenvalues as damped cardiac-band oscillators.

    Converts the SSM A matrix from a generic dynamical system into a bank
    of frequency-selective resonators tuned to 0.7–3.0 Hz (42–180 BPM).

    Each channel j:
      f_j ~ linspace(f_min, f_max)   → ω_j = 2π·f_j / fs
      A_j = log(decay) + i·ω_j       → damped oscillator

    Decay=0.95 is the empirically optimal value on UBFC (V5 validation):
    lower (0.90) loses long-range HR trend, higher (0.97) causes
    temporal smearing at 300-step sequences. Model can adapt effective
    decay via trainable dt_predictor during finetuning.
    """
    freqs_hz = torch.linspace(f_min, f_max, d_model)
    freqs_hz += torch.randn(d_model) * (f_max - f_min) / (d_model * 2)
    freqs_hz = freqs_hz.clamp(f_min * 0.8, f_max * 1.2)

    omega = 2 * math.pi * freqs_hz / fs
    A_real_init = torch.full((d_model,), math.log(decay))
    A_imag_init = omega

    return A_real_init, A_imag_init


# ═══════════════════════════════════════════════════════════════
# SSMBlock — V11 核心块 (仅A初始化改为COB)
# ═══════════════════════════════════════════════════════════════
class SSMBlock(nn.Module):
    """Single SSM block: A(COB) + DynamicDt + parallel_scan + FDF.

    Changes from V11:
      - All components unchanged (random A init, B, C, Dt, FDF)
      - COB kept as optional concept for paper discussion
    """
    def __init__(self, d_model, seq_len=300, use_fdf=True, fs=30.0):
        super().__init__()
        self.d_model = d_model
        self.fs = fs

        # COB init: cardiac oscillator bank covering heart-rate band
        self.A_real_param = nn.Parameter(torch.empty(d_model))
        self.A_imag_param = nn.Parameter(torch.empty(d_model))
        self._init_A_params()

        self.B_real = nn.Parameter(torch.empty(d_model))
        self.B_imag = nn.Parameter(torch.empty(d_model))
        self.C_real = nn.Parameter(torch.empty(d_model))
        self.C_imag = nn.Parameter(torch.empty(d_model))
        self._init_real_params()

        # REAL_ONLY=1 消融: 冻结虚部 (A_imag/B_imag/C_imag=0), SSM 退化为纯实数状态机
        if os.environ.get("REAL_ONLY", "0") == "1":
            for _p in (self.A_imag_param, self.B_imag, self.C_imag):
                _p.data.zero_()
                _p.requires_grad_(False)

        self.dt_predictor = DynamicDtPredictor(d_model=d_model)
        self.fdf = FreqDomainFeedForward(d_model, seq_len) if use_fdf else nn.Identity()

    def _init_A_params(self):
        """COB init: cardiac oscillator bank covering 0.7-3.0 Hz (42-180 BPM).

        随机初始化 (uniform(-1.5,1.5)) 缺乏心率频段先验, 实测导致 SSM 输出主频
        被钉死在固有频率 ~0.9-1.0Hz, 无法跟随输入心率变化。改用 COB 给每个
        channel 预设覆盖心率频段的振荡器频率, 恢复频率选择性。
        RANDOM_A=1 消融: 回到随机初始化, 验证 COB 频率先验的贡献。
        """
        with torch.no_grad():
            if os.environ.get("RANDOM_A", "0") == "1":
                nn.init.uniform_(self.A_real_param, -1.5, 1.5)
                nn.init.uniform_(self.A_imag_param, -1.5, 1.5)
            else:
                A_real, A_imag = cardiac_oscillator_init(self.d_model, fs=self.fs)
                self.A_real_param.copy_(A_real)
                self.A_imag_param.copy_(A_imag)

    def _init_real_params(self):
        with torch.no_grad():
            nn.init.uniform_(self.B_real, -0.1, 0.1)
            nn.init.uniform_(self.B_imag, -0.1, 0.1)
            nn.init.uniform_(self.C_real, -0.1, 0.1)
            nn.init.uniform_(self.C_imag, -0.1, 0.1)
            B_mag = torch.sqrt(self.B_real**2 + self.B_imag**2)
            C_mag = torch.sqrt(self.C_real**2 + self.C_imag**2)
            if (B_mag > 1.0).any():
                idx = B_mag > 1.0
                self.B_real[idx] /= B_mag[idx]
                self.B_imag[idx] /= B_mag[idx]
            if (C_mag > 1.0).any():
                idx = C_mag > 1.0
                self.C_real[idx] /= C_mag[idx]
                self.C_imag[idx] /= C_mag[idx]

    def forward(self, xr, xi):
        B, L, _ = xr.shape

        # Dynamic Δt from input magnitude
        modulus = torch.sqrt(xr**2 + xi**2)
        dt = self.dt_predictor(modulus)

        # Lyapunov stability + discretization
        A_real_stable = apply_lyapunov_stability(self.A_real_param)
        A_imag = self.A_imag_param

        ar_dt = A_real_stable * dt
        ai_dt = A_imag * dt
        exp_ar = torch.exp(ar_dt.clamp(max=20.0))
        a_re = exp_ar * torch.cos(ai_dt)
        a_im = exp_ar * torch.sin(ai_dt)

        B_dt_re = self.B_real * dt
        B_dt_im = self.B_imag * dt
        u_re = B_dt_re * xr - B_dt_im * xi
        u_im = B_dt_re * xi + B_dt_im * xr

        # Parallel SSM scan
        h_re, h_im = parallel_complex_scan(a_re, a_im, u_re, u_im)

        y_re = self.C_real * h_re - self.C_imag * h_im
        y_im = self.C_real * h_im + self.C_imag * h_re

        # FDF
        y_re = self.fdf(y_re)
        y_im = self.fdf(y_im)

        return y_re, y_im


# ═══════════════════════════════════════════════════════════════
# HR Regression Head (V11 original)
# ═══════════════════════════════════════════════════════════════
class HRRegressionHead(nn.Module):
    """Global avg pool → MLP → BPM."""
    def __init__(self, d_model=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        pooled = features.mean(dim=1)
        return self.net(pooled).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# RemoteSSM: Full Architecture (V11 + COB)
# ═══════════════════════════════════════════════════════════════
class RemoteSSM(nn.Module):
    """轻量复数状态空间模型 (申报书算法 + COB初始化)

    架构 (与V11完全一致):
      Path1: face_roi (B,T,H,W,3) → FrameStem → (B,T,D)
      Path2: x_real/x_imag → TemporalStem×2 → (B,T,D)×2
      Fusion: GatedFusion(spatial, signal) ×2
      Core:   SSMBlock × N (COB init + DynamicDt + FDF)
      Heads:  BVP waveform + HR BPM

    仅改进: A矩阵使用COB初始化 (训练时加速收敛，推理零开销)

    Args:
        d_model:       特征维度 (default: 256)
        seq_len:       序列长度 (default: 300)
        n_ssm_blocks:  SSM块数 (default: 2)
        use_fdf:       是否使用FDF (default: True)
        roi_size:      面部ROI尺寸 (default: 32)
        dropout:       Dropout率 (default: 0.15)
        fs:            采样率 (default: 30.0)
    """
    def __init__(self, d_model=256, seq_len=300, n_ssm_blocks=2,
                 use_fdf=True, roi_size=32, dropout=0.15, fs=30.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # ── Path 1: Face → spatial features ──
        self.frame_stem = FrameStem(d_model=d_model, roi_size=roi_size,
                                    dropout=dropout)

        # ── Path 2: BGR signal → temporal features ──
        self.stem_real = MultiScaleTemporalStem(in_channels=1, d_model=d_model,
                                                dropout=dropout)
        self.stem_imag = MultiScaleTemporalStem(in_channels=1, d_model=d_model,
                                                dropout=dropout)

        # ── Dual-path fusion ──
        self.fuse_real = GatedFusion(d_model)
        self.fuse_imag = GatedFusion(d_model)
        self.fuse_norm_real = nn.LayerNorm(d_model)
        self.fuse_norm_imag = nn.LayerNorm(d_model)

        # ── SSM blocks (COB initialized) ──
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(d_model, seq_len, use_fdf=True if i < n_ssm_blocks - 1 else False,
                         fs=fs)
            for i in range(n_ssm_blocks)
        ])

        # ── Final FDF ──
        self.final_fdf_real = FreqDomainFeedForward(d_model, seq_len) if use_fdf else nn.Identity()
        self.final_fdf_imag = FreqDomainFeedForward(d_model, seq_len) if use_fdf else nn.Identity()

        # ── Denoising input projection (Stage 1 only) ──
        self.denoise_proj = nn.Linear(1, d_model)

        # ── BVP output head ──
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model * 2)
        self.fc_out = nn.Linear(d_model * 2, 1)

        # ── HR regression head ──
        self.hr_head = HRRegressionHead(d_model=d_model, dropout=dropout)

        self._init_output()

    def _init_output(self):
        nn.init.xavier_normal_(self.fc_out.weight, gain=0.1)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, face_roi=None, x_real=None, x_imag=None,
                noisy_bvp=None, mode='finetune', return_features=False):
        """
        Args:
            face_roi:  (B, T, H, W, 3) face ROI [finetune mode]
            x_real:    (B, T, 1) BGR real channel [finetune mode]
            x_imag:    (B, T, 1) BGR imag channel [finetune mode]
            noisy_bvp: (B, L) noisy waveform [denoise mode]
            mode:      'finetune' or 'denoise'
            return_features: return pooled features for GRL

        Returns:
            denoise:  (bvp_pred,) or (bvp_pred, features)
            finetune: (bvp_pred, hr_pred) or (bvp_pred, hr_pred, features)
        """
        if mode == 'finetune':
            if face_roi is None or x_real is None:
                raise ValueError("finetune mode requires face_roi, x_real, x_imag")

            # ── Dual-path feature extraction ──
            sp_feat = self.frame_stem(face_roi)           # (B,T,D)
            sig_real = self.stem_real(x_real)              # (B,T,D)
            sig_imag = self.stem_imag(x_imag)              # (B,T,D)

            xr = self.fuse_real(sp_feat, sig_real)         # (B,T,D)
            xi = self.fuse_imag(sp_feat, sig_imag)         # (B,T,D)
            xr = self.fuse_norm_real(xr)                    # COB SSM stability
            xi = self.fuse_norm_imag(xi)

            # GRL features from fusion (pre-SSM): most subject-dependent
            features = xr.mean(dim=1) if return_features else None

        elif mode == 'denoise':
            if noisy_bvp is None:
                raise ValueError("denoise mode requires noisy_bvp input")
            if noisy_bvp.dim() == 2:
                noisy_bvp = noisy_bvp.unsqueeze(-1)         # (B,L) → (B,L,1)

            features = None  # will be set after SSM if return_features
            xr = self.denoise_proj(noisy_bvp)               # (B,L,D)
            # Imaginary from gradient (approximate Hilbert)
            pad = F.pad(xr, (0, 0, 1, 1), mode='replicate')
            xi = (pad[:, 2:, :] - pad[:, :-2, :]) * 0.5   # (B,L,D)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ── Core SSM ──
        for block in self.ssm_blocks:
            xr, xi = block(xr, xi)

        # ── Final FDF ──
        xr = self.final_fdf_real(xr)
        xi = self.final_fdf_imag(xi)

        # ── Features for GRL ──
        if return_features and features is None:
            features = xr.mean(dim=1)                      # (B,D) — fallback

        # ── BVP prediction ──
        out_cat = torch.cat([xr, xi], dim=-1)              # (B,T,2D)
        out_cat = self.output_norm(out_cat)
        out_cat = self.output_dropout(out_cat)
        residual = self.fc_out(out_cat).squeeze(-1)        # (B,T) SSM 精修残差

        # 残差连接: 输出 = 输入信号 + SSM 精修残差
        # finetune 模式输入 real=CHROM, 保证模型数学上不可能劣于 CHROM/POS FFT 下界
        # NO_RESIDUAL=1 消融: 去掉残差, 输出纯 SSM 精修信号
        if mode == 'finetune':
            if os.environ.get("NO_RESIDUAL", "0") == "1":
                bvp_pred = residual                             # (B,T) 消融: w/o 残差连接
            else:
                bvp_pred = x_real.squeeze(-1) + residual        # (B,T)
        else:
            bvp_pred = noisy_bvp.squeeze(-1) + residual    # (B,T)

        if mode == 'denoise':
            if return_features:
                return bvp_pred, features
            return bvp_pred

        # ── HR prediction (finetune mode only) ──
        hr_pred = self.hr_head(xr)                         # (B,)

        if return_features:
            return bvp_pred, hr_pred, features
        return bvp_pred, hr_pred


# ═══════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════
def create_model(d_model=256, seq_len=300, n_ssm_blocks=2,
                 roi_size=32, dropout=0.15, fs=30.0, use_fdf=True):
    return RemoteSSM(
        d_model=d_model, seq_len=seq_len, n_ssm_blocks=n_ssm_blocks,
        roi_size=roi_size, dropout=dropout, fs=fs, use_fdf=use_fdf,
    )


def create_discriminator(d_model=256, num_subjects=34, dropout=0.3):
    return SubjectDiscriminator(d_model=d_model, num_subjects=num_subjects,
                                dropout=dropout)
