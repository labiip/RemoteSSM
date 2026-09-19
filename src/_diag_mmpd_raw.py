"""MMPD 原始视频信号质量验证.

直接读 .mat 原始 video + GT_ppg, 用整帧均值(不依赖人脸检测)提取信号,
验证 MMPD 视频里脉搏信息是否存在, 排除 ROI 检测干扰.

用法: python3 _diag_mmpd_raw.py <mat_path> [<mat_path> ...]
"""
import sys
import numpy as np
import scipy.io

FS = 30.0
LAGS = range(-60, 61)


def corr_at_lag(a, b, lag):
    if lag > 0:
        x, y = a[:len(b) - lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:len(b) + lag]
    else:
        x, y = a, b
    n = min(len(x), len(y))
    x, y = x[:n].astype(np.float64), y[:n].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    for path in sys.argv[1:]:
        try:
            d = scipy.io.loadmat(path, variable_names=["video", "GT_ppg"])
            video = d["video"]                      # (T,H,W,C) RGB [0,1]
            ppg = np.asarray(d["GT_ppg"], dtype=np.float32).reshape(-1)
        except Exception as e:
            print(f"[{path}] load fail: {e}")
            continue
        T = min(video.shape[0], len(ppg))
        v = video[:T]
        ppg_raw = ppg[:T].astype(np.float64)
        ppg = (ppg_raw - ppg_raw.mean()) / (ppg_raw.std() + 1e-8)
        # 整帧均值 RGB
        mean_rgb = v.mean(axis=(1, 2))              # (T,3) R,G,B
        g = mean_rgb[:, 1]
        # CHROM (RGB 序)
        R, G, B = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        X = R - G
        Y = 0.5 * R + 0.5 * G - B
        X = X - X.mean(); Y = Y - Y.mean()
        a = X.std() / (Y.std() + 1e-8)
        chrom = X - a * Y
        # 每通道 lag 扫描 vs ppg
        print(f"[{path.split('/')[-1]}] T={T}")
        for name, sig in [("G", g), ("CHROM", chrom)]:
            s = sig.astype(np.float64)
            s = (s - s.mean()) / (s.std() + 1e-8)
            rs = []
            for lag in LAGS:
                rs.append(abs(corr_at_lag(s, ppg, lag)))
            bi = int(np.argmax(rs))
            print(f"  {name:6s}: lag_best={LAGS[bi]:+d} r={rs[bi]:.3f} | r(lag0)={abs(corr_at_lag(s, ppg, 0)):.3f}")


if __name__ == "__main__":
    main()
