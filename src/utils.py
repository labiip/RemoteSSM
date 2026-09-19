import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class DifferentiablePhysioExtractor(nn.Module):
    # 基于全序列 1D RFFT + 自适应汉宁窗 + 频谱去底噪
    # 彻底解决短窗 STFT 导致的低频(呼吸)泄漏，以及平坦底噪造成的心率期望值偏移
    def __init__(self, fs=30.0):
        super().__init__()
        self.fs = fs

    def forward(self, bvp_signals, freq_range=(0.7, 3.0), dynamic_fs=None,
                temperature=0.05):
        """
        可微分生理参数提取器。

        Args:
            bvp_signals:  (B, L) BVP 波形
            freq_range:   目标频段 (Hz)
            dynamic_fs:   动态采样率，None 则使用构造函数中的 fs
            temperature:  Softmax 温度参数:
                          - 0.05 (默认): 极度锐利，精准剥离最高波峰
                          - 0.5:         训练早期用，梯度更平滑稳定
                          - 1.0:         软注意力，多峰融合

        Returns:
            hr_bpm: (B,) 心率预测值
        """
        fs = float(dynamic_fs) if dynamic_fs is not None else self.fs
        seq_len = bvp_signals.shape[-1]
        device = bvp_signals.device
        dtype = bvp_signals.dtype

        # 1. 信号去均值与归一化
        bvp_zero = bvp_signals - bvp_signals.mean(dim=-1, keepdim=True)
        bvp_norm = bvp_zero / (bvp_zero.std(dim=-1, keepdim=True) + 1e-8)

        # 2. 全序列汉宁窗
        window = torch.hann_window(seq_len, device=device, dtype=dtype)
        bvp_windowed = bvp_norm * window

        # 3. 高分辨率 FFT (补零至 ≥1024 点，频率分辨率 ~0.029 Hz)
        n_fft = max(1024, 1 << math.ceil(math.log2(seq_len)))
        # AMP (FP16) 保护: rfft 在 FP32 下数值更稳定
        bvp_fft_in = bvp_windowed.float() if bvp_windowed.dtype != torch.float32 else bvp_windowed
        fft_out = torch.fft.rfft(bvp_fft_in, n=n_fft, dim=-1)
        power_spectrum = torch.abs(fft_out) ** 2  # (B, N_fft//2+1)

        # 频率轴
        freqs = torch.linspace(0, fs / 2, n_fft // 2 + 1,
                               device=device, dtype=dtype)

        # 4. 提取目标生理频段
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not torch.any(freq_mask):
            return torch.full((bvp_signals.shape[0],), 0.0,
                              device=device, dtype=dtype)

        valid_freqs = freqs[freq_mask]                     # (M,)
        power_masked = power_spectrum[:, freq_mask]         # (B, M)

        # 5. 频谱去底噪
        noise_floor = power_masked.mean(dim=-1, keepdim=True)
        power_clean = F.relu(power_masked - noise_floor)

        # 6. Softmax 加权频率重心
        #    使用入参 temperature 替代硬编码，支持训练时平滑/推理时锐利
        power_norm = power_clean / (power_clean.max(dim=-1, keepdim=True)[0] + 1e-8)
        weights = F.softmax(power_norm / max(temperature, 1e-4), dim=-1)
        pred_hr_hz = torch.sum(weights * valid_freqs.unsqueeze(0), dim=-1)

        return pred_hr_hz * 60.0


class KalmanFilter1D:
    # 卡尔曼滤波器，用于平滑推理时的输出
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement = 0.0
        self.posteri_error_estimate = 1.0
        self.is_initialized = False

    def update(self, measurement):
        if not self.is_initialized:
            self.estimated_measurement = measurement
            self.is_initialized = True
            return self.estimated_measurement

        priori_estimate = self.estimated_measurement
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (
            priori_error_estimate + self.measurement_variance
        )
        self.estimated_measurement = priori_estimate + blending_factor * (
            measurement - priori_estimate
        )
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.estimated_measurement


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except (RuntimeError, pickle.UnpicklingError, EOFError, ValueError):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)