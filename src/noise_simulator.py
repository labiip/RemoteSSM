"""
PhysioNoise Simulator: Physiologically-motivated PPG noise generator.

Generates 6 classes of realistic noise that mimic real-world rPPG degradation:
  1. Motion Artifact    — amplitude modulation + frequency shift (head movement)
  2. Illumination Drift — slow baseline wander (lighting changes, clouds)
  3. Sensor Shot Noise  — Poisson-Gaussian mixture (camera CMOS noise)
  4. Compression Artifact — quantization + blocking (video codec)
  5. Environmental EMI  — 50/60Hz mains + respiratory band (0.1–0.5 Hz)
  6. Frame Dropout      — random temporal masking (occlusion, tracker loss)

Designed to produce paired (noisy, clean) training data for denoising pretraining.
Each clean BVP waveform can generate unlimited noisy variants with randomized
parameters drawn from physiologically-plausible ranges.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1. Motion Artifact: Amplitude + Frequency Jitter
# ═══════════════════════════════════════════════════════════════
def motion_artifact(bvp, fs=30.0, max_amp_mod=0.35, max_freq_shift=0.08):
    """Simulate head-motion-induced amplitude/frequency modulation.

    Real motion causes:
      - Amplitude modulation (distance changes → signal strength varies)
      - Frequency jitter (Doppler-like effect from movement relative to light)

    Args:
        bvp: (B, L) clean BVP waveform
        fs: sampling rate (Hz)
        max_amp_mod: maximum amplitude modulation depth [0, 1]
        max_freq_shift: maximum frequency shift as fraction of fs

    Returns:
        bvp_modulated: (B, L) signal with motion artifact
    """
    B, L = bvp.shape
    device = bvp.device

    # Amplitude modulation: random low-frequency envelope
    env_freq = torch.FloatTensor(B).uniform_(0.05, 0.8).to(device)  # 0.05–0.8 Hz drift
    env_phase = torch.FloatTensor(B).uniform_(0, 2 * np.pi).to(device)
    amp_mod = torch.FloatTensor(B).uniform_(0.05, max_amp_mod).to(device)

    t = torch.arange(L, device=device, dtype=torch.float32) / fs
    t = t.unsqueeze(0)  # (1, L)

    # Multi-sine envelope for realistic shape
    envelope = 1.0 + amp_mod.unsqueeze(1) * (
        torch.sin(2 * np.pi * env_freq.unsqueeze(1) * t + env_phase.unsqueeze(1))
        + 0.3 * torch.sin(4 * np.pi * env_freq.unsqueeze(1) * t + env_phase.unsqueeze(1) * 1.7)
    )

    # Frequency jitter via phase modulation
    jitter_freq = torch.FloatTensor(B).uniform_(0.2, 1.5).to(device)
    jitter_depth = torch.FloatTensor(B).uniform_(0.02, max_freq_shift).to(device)
    jitter_phase = torch.FloatTensor(B).uniform_(0, 2 * np.pi).to(device)

    phase_jitter = jitter_depth.unsqueeze(1) * torch.sin(
        2 * np.pi * jitter_freq.unsqueeze(1) * t + jitter_phase.unsqueeze(1)
    )

    # Apply via FFT with phase shift
    bvp_fft = torch.fft.rfft(bvp.float(), dim=1)
    n_freq = bvp_fft.shape[1]
    freq_idx = torch.arange(n_freq, device=device, dtype=torch.float32)
    phase_shift = phase_jitter[:, :n_freq]

    # Modulate amplitude and phase in frequency domain
    mag, phase_ang = torch.abs(bvp_fft), torch.angle(bvp_fft)
    phase_ang = phase_ang + phase_shift * torch.randn(B, 1, device=device) * 0.3
    bvp_mod_fft = mag * envelope[:, :n_freq] * torch.exp(1j * phase_ang)
    bvp_mod = torch.fft.irfft(bvp_mod_fft, n=L, dim=1)

    return bvp_mod.to(bvp.dtype)


# ═══════════════════════════════════════════════════════════════
# 2. Illumination Drift: Slow Baseline Wander
# ═══════════════════════════════════════════════════════════════
def illumination_drift(bvp, fs=30.0, max_drift_amp=0.25):
    """Simulate slow lighting changes (clouds, indoor light flicker).

    Adds low-frequency baseline wander (<0.5 Hz) that mimics:
      - Clouds passing → slow brightness change
      - Fluorescent flicker → 100/120 Hz harmonics aliased to low freq
      - Auto-exposure adjustments

    Args:
        bvp: (B, L)
        fs: sampling rate
        max_drift_amp: maximum drift amplitude relative to signal std

    Returns:
        bvp_with_drift: (B, L)
    """
    B, L = bvp.shape
    device = bvp.device

    signal_std = bvp.float().std(dim=1, keepdim=True) + 1e-6

    # Multiple low-frequency trends
    drift = torch.zeros(B, L, device=device, dtype=torch.float32)
    t = torch.arange(L, device=device, dtype=torch.float32) / fs

    for _ in range(3):
        freq = torch.FloatTensor(B).uniform_(0.005, 0.4).to(device)
        phase = torch.FloatTensor(B).uniform_(0, 2 * np.pi).to(device)
        amp = torch.FloatTensor(B).uniform_(0.05, max_drift_amp).to(device)
        drift += amp.unsqueeze(1) * signal_std * torch.sin(
            2 * np.pi * freq.unsqueeze(1) * t.unsqueeze(0) + phase.unsqueeze(1)
        )

    # Add occasional step changes (camera auto-exposure adjustment)
    do_step = torch.FloatTensor(B).uniform_(0, 1) > 0.3
    if do_step.any():
        step_pos = torch.randint(L // 4, 3 * L // 4, (B,), device=device)
        step_val = (signal_std * torch.FloatTensor(B, 1).uniform_(-max_drift_amp, max_drift_amp).to(device)).squeeze(-1)
        step_mask = (t.unsqueeze(0) >= step_pos.unsqueeze(1) / fs).float()
        drift = drift + step_val.unsqueeze(1) * step_mask * do_step.float().unsqueeze(1)

    return (bvp.float() + drift).to(bvp.dtype)


# ═══════════════════════════════════════════════════════════════
# 3. Sensor Shot Noise: Poisson-Gaussian Mixture
# ═══════════════════════════════════════════════════════════════
def sensor_noise(bvp, gain_mean=0.015, gain_std=0.005, read_noise_std=0.003):
    """Simulate CMOS sensor noise (Poisson + Gaussian).

    Camera noise model (standard in CV):
      N = N_shot ~ Poisson(signal * gain) + N_read ~ Gaussian(0, σ²_read)

    Args:
        bvp: (B, L)
        gain_mean: mean Poisson gain (photon shot noise)
        gain_std: std of gain across batches
        read_noise_std: readout noise std

    Returns:
        bvp_noisy: (B, L)
    """
    B, L = bvp.shape
    device = bvp.device

    # Normalize to [0, 1] range for Poisson simulation
    bvp_norm = (bvp.float() - bvp.float().min())
    bvp_range = bvp_norm.max() + 1e-8
    bvp_norm = bvp_norm / bvp_range

    # Poisson shot noise (signal-dependent)
    gain = torch.FloatTensor(B).uniform_(gain_mean - gain_std, gain_mean + gain_std).to(device)
    # Rate = signal * (1/gain), clipped to non-negative
    rate = bvp_norm / (gain.unsqueeze(1) + 1e-8)
    rate = rate.clamp(min=1e-6)
    shot_noise = (torch.poisson(rate) - rate) * gain.unsqueeze(1)

    # Gaussian read noise
    read_noise = torch.randn(B, L, device=device) * read_noise_std

    bvp_noisy = bvp_norm + shot_noise + read_noise
    bvp_noisy = bvp_noisy * bvp_range + bvp.float().min()

    return bvp_noisy.to(bvp.dtype)


# ═══════════════════════════════════════════════════════════════
# 4. Compression Artifact: Quantization + Blocking
# ═══════════════════════════════════════════════════════════════
def compression_artifact(bvp, q_levels=None, block_size_range=(8, 32)):
    """Simulate video compression artifacts (H.264/HEVC-like).

    Effects:
      - Quantization: signal values rounded to discrete levels
      - Blocking: temporal blocks get correlated distortion

    Args:
        bvp: (B, L)
        q_levels: number of quantization levels (None = random per batch)
        block_size_range: (min, max) temporal block size for blocking artifacts

    Returns:
        bvp_compressed: (B, L)
    """
    B, L = bvp.shape
    device = bvp.device

    if q_levels is None:
        q_levels = torch.randint(16, 128, (B,), device=device).float()

    bvp_f = bvp.float()
    signal_range = bvp_f.max(dim=1, keepdim=True).values - bvp_f.min(dim=1, keepdim=True).values

    # Quantization
    q_step = signal_range / q_levels.unsqueeze(1).clamp(min=1)
    bvp_quant = torch.round(bvp_f / q_step) * q_step

    # Blocking: random temporal blocks get extra distortion
    block_size = torch.randint(block_size_range[0], block_size_range[1], (B,), device=device)
    for b in range(B):
        n_blocks = L // block_size[b].item()
        if n_blocks < 2:
            continue
        affected_blocks = torch.rand(n_blocks, device=device) > 0.6
        if affected_blocks.any():
            block_noise = torch.randn(1, device=device) * signal_range[b] * 0.08
            for j in range(n_blocks):
                if affected_blocks[j]:
                    start, end = j * block_size[b].item(), min((j + 1) * block_size[b].item(), L)
                    bvp_quant[b, start:end] += block_noise

    return bvp_quant.to(bvp.dtype)


# ═══════════════════════════════════════════════════════════════
# 5. Environmental EMI: Powerline + Respiratory Interference
# ═══════════════════════════════════════════════════════════════
def environmental_emi(bvp, fs=30.0):
    """Simulate environmental electromagnetic interference.

    - 50/60 Hz power line noise (aliased to low freq by 30Hz sampling)
    - 0.1–0.5 Hz respiratory band (breathing modulates BVP amplitude)

    Args:
        bvp: (B, L)
        fs: sampling rate

    Returns:
        bvp_noisy: (B, L)
    """
    B, L = bvp.shape
    device = bvp.device

    signal_std = bvp.float().std(dim=1, keepdim=True) + 1e-6
    t = torch.arange(L, device=device, dtype=torch.float32) / fs
    noise = torch.zeros(B, L, device=device, dtype=torch.float32)

    # Aliased 50 Hz: 50 mod 30 = 20 Hz alias; 60 mod 30 = 30 (Nyquist) / 0 Hz alias
    mains_freqs = [10.0, 20.0]  # common aliased frequencies at 30Hz sampling
    for freq in mains_freqs:
        amp = torch.FloatTensor(B).uniform_(0.005, 0.02).to(device)
        phase = torch.FloatTensor(B).uniform_(0, 2 * np.pi).to(device)
        noise += amp.unsqueeze(1) * signal_std * torch.sin(
            2 * np.pi * freq * t.unsqueeze(0) + phase.unsqueeze(1)
        )

    # Respiratory modulation (amplitude modulation at 0.1–0.5 Hz)
    resp_freq = torch.FloatTensor(B).uniform_(0.1, 0.5).to(device)
    resp_amp = torch.FloatTensor(B).uniform_(0.1, 0.3).to(device)
    resp_phase = torch.FloatTensor(B).uniform_(0, 2 * np.pi).to(device)
    resp_mod = 1.0 + resp_amp.unsqueeze(1) * torch.sin(
        2 * np.pi * resp_freq.unsqueeze(1) * t.unsqueeze(0) + resp_phase.unsqueeze(1)
    )

    # Respiratory additive component
    resp_add = torch.FloatTensor(B).uniform_(0.02, 0.08).to(device)
    resp_noise = resp_add.unsqueeze(1) * signal_std * torch.sin(
        2 * np.pi * resp_freq.unsqueeze(1) * t.unsqueeze(0)
    )

    return (bvp.float() * resp_mod + noise + resp_noise).to(bvp.dtype)


# ═══════════════════════════════════════════════════════════════
# 6. Frame Dropout: Random Temporal Masking
# ═══════════════════════════════════════════════════════════════
def frame_dropout(bvp, max_mask_len=15, max_num_masks=5):
    """Simulate frame loss from face tracker failure or occlusion.

    Random contiguous segments are zeroed out, mimicking:
      - Face detection failures (person turns away)
      - Hand/object occlusion
      - Motion blur → tracker lost

    The signal is linearly interpolated across gaps for continuity.

    Args:
        bvp: (B, L)
        max_mask_len: maximum consecutive masked frames
        max_num_masks: maximum number of mask segments per batch

    Returns:
        bvp_masked: (B, L)
        mask: (B, L) binary mask (1 = kept, 0 = dropped)
    """
    B, L = bvp.shape
    device = bvp.device

    mask = torch.ones(B, L, device=device, dtype=bvp.dtype)

    for b in range(B):
        num_masks = torch.randint(1, max_num_masks + 1, (1,)).item()
        for _ in range(num_masks):
            mask_len = torch.randint(2, max_mask_len + 1, (1,)).item()
            if mask_len >= L - 2:
                continue
            start = torch.randint(0, L - mask_len, (1,)).item()
            mask[b, start:start + mask_len] = 0.0

    # Linear interpolation across gaps
    bvp_f = bvp.float()
    bvp_masked = bvp_f * mask
    kernel = torch.ones(1, 1, 7, device=device) / 7
    # Apply interpolation via convolution on gaps
    for b in range(B):
        gaps = mask[b] == 0
        if gaps.any() and not gaps.all():
            bvp_np = bvp_masked[b].cpu().numpy()
            mask_np = mask[b].cpu().numpy()

            # Find gap boundaries and interpolate
            gap_starts = []
            gap_ends = []
            in_gap = False
            for i in range(L):
                if mask_np[i] == 0 and not in_gap:
                    gap_starts.append(i)
                    in_gap = True
                elif mask_np[i] == 1 and in_gap:
                    gap_ends.append(i)
                    in_gap = False
            if in_gap:
                gap_ends.append(L)

            for start, end in zip(gap_starts, gap_ends):
                if start > 0 and end < L:
                    v_start = bvp_np[start - 1]
                    v_end = bvp_np[end]
                    gap_len = end - start
                    for k in range(1, gap_len + 1):
                        alpha = k / (gap_len + 1)
                        bvp_np[start + k - 1] = v_start * (1 - alpha) + v_end * alpha

            bvp_masked[b] = torch.tensor(bvp_np, device=device, dtype=torch.float32)

    return bvp_masked.to(bvp.dtype), mask


# ═══════════════════════════════════════════════════════════════
# Combined Noise Pipeline
# ═══════════════════════════════════════════════════════════════

class PhysioNoiseSimulator:
    """Combined PPG noise simulation pipeline.

    Produces (noisy, clean) pairs for denoising pretraining.
    Each noise type is independently applied with configurable probability.

    Usage:
        sim = PhysioNoiseSimulator(fs=30.0)
        clean_bvp = torch.randn(16, 300)  # (B, L) from dataset
        noisy_bvp = sim(clean_bvp)         # (B, L) with randomized noise
    """
    def __init__(self, fs=30.0, noise_probs=None, device='cpu'):
        self.fs = fs
        self.device = device

        # Default noise type probabilities
        self.noise_probs = noise_probs or {
            'motion':        0.9,
            'illumination':  0.8,
            'sensor':        1.0,   # always present in real cameras
            'compression':   0.6,
            'emi':           0.5,
            'dropout':       0.4,
        }

    def __call__(self, bvp_clean, return_mask=False):
        """Apply random noise combination to clean BVP.

        Args:
            bvp_clean: (B, L) clean BVP waveform
            return_mask: if True, also return the dropout mask

        Returns:
            bvp_noisy: (B, L) corrupted signal
            mask: (B, L) optional dropout mask
        """
        B = bvp_clean.shape[0]
        bvp = bvp_clean.clone()
        dropout_mask = None

        # Apply noise types in physiologically-correct order:
        # 1. Illumination first (changes baseline)
        # 2. Motion (modulates amplitude/frequency)
        # 3. EMI (modulates + adds interference)
        # 4. Sensor noise (additive, signal-dependent)
        # 5. Compression (quantization after analog noise)
        # 6. Dropout (temporal masking)

        if torch.rand(1).item() < self.noise_probs['illumination']:
            drift_amp = 0.15 + torch.rand(1).item() * 0.15  # 0.15–0.30
            bvp = illumination_drift(bvp, self.fs, max_drift_amp=drift_amp)

        if torch.rand(1).item() < self.noise_probs['motion']:
            amp_mod = 0.15 + torch.rand(1).item() * 0.25  # 0.15–0.40
            freq_shift = 0.03 + torch.rand(1).item() * 0.07  # 0.03–0.10
            bvp = motion_artifact(bvp, self.fs, max_amp_mod=amp_mod, max_freq_shift=freq_shift)

        if torch.rand(1).item() < self.noise_probs['emi']:
            bvp = environmental_emi(bvp, self.fs)

        if torch.rand(1).item() < self.noise_probs['sensor']:
            bvp = sensor_noise(bvp)

        if torch.rand(1).item() < self.noise_probs['compression']:
            bvp = compression_artifact(bvp)

        if torch.rand(1).item() < self.noise_probs['dropout']:
            bvp, dropout_mask = frame_dropout(bvp)

        if return_mask:
            return bvp, dropout_mask
        return bvp


def create_denoising_batch(bvp_batch, simulator, fs=30.0):
    """Create paired (noisy_input, clean_target) for training.

    Args:
        bvp_batch: (B, L) clean BVP waveforms from dataset
        simulator: PhysioNoiseSimulator instance
        fs: sampling rate

    Returns:
        noisy: (B, L) corrupted signals
        clean: (B, L) original clean signals
    """
    clean = bvp_batch.clone()
    noisy = simulator(clean)
    return noisy, clean
