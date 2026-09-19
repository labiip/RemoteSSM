"""BVP 波形质量验证: FFT 主峰是否在心率带 (0.7-3.0 Hz).

验证 MMPD 的 GT_ppg 是否包含清晰心率信号.
用法: python3 _diag_bvp_quality.py <mat_or_txt_path> [fs]
"""
import sys
import numpy as np

FS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
LO, HI = 0.7, 3.0


def analyze(name, ppg):
    ppg = np.asarray(ppg, dtype=np.float64)
    T = len(ppg)
    ppg = ppg - ppg.mean()
    s = ppg / (ppg.std() + 1e-8)
    sw = s * np.hanning(T)
    nfft = max(2048, 1 << int(np.ceil(np.log2(T))))
    spec = np.abs(np.fft.rfft(sw, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / FS)
    m = (freqs >= LO) & (freqs <= HI)
    in_band = spec[m].sum() / (spec.sum() + 1e-12)
    pk = freqs[m][np.argmax(spec[m])]
    # 主峰锐度: 主峰能量 / 带内平均
    sharp = spec[m].max() / (spec[m].mean() + 1e-12)
    print(f"{name}: T={T} | 带内能量比={in_band:.1%} | 主峰={pk*60:.1f} BPM | 锐度={sharp:.1f}")


def main():
    for path in sys.argv[1:]:
        if path.endswith(".mat"):
            import scipy.io
            d = scipy.io.loadmat(path, variable_names=["GT_ppg"])
            ppg = np.asarray(d["GT_ppg"], dtype=np.float32).reshape(-1)
            analyze(path.split("/")[-1], ppg)
        else:
            gt = np.loadtxt(path)
            if gt.ndim == 2 and gt.shape[1] >= 3:
                bvp = gt[:, 0]
            elif gt.ndim == 2 and gt.shape[0] == 3:
                bvp = gt[0]
            else:
                bvp = gt.flatten()
            analyze(path.split("/")[-1], bvp)


if __name__ == "__main__":
    main()
