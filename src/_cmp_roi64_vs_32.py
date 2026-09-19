"""64x64 vs 32x32 预处理信息量对比 (同一视频 p1_0).

旧 32x32 (combined/subject51_a0, 旧tracker) vs 新 64x64 (test64, 新tracker).
比较: 整体G均 + 8x8子块带通FFT HR vs GT BVP HR.
用法: python3 _cmp_roi64_vs_32.py
"""
import os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 30.0
SEQ = 300
STEP = 30
LO, HI = 0.7, 3.0
N_SUB = 8


def butter_bandpass(n=1, lo=LO, hi=HI, fs=FS):
    return butter(n, [lo, hi], btype="band", fs=fs, output="sos")


def fft_hr(sig):
    sig = np.asarray(sig, np.float32)
    if len(sig) < 16:
        return np.nan
    sig = sig - sig.mean()
    s = sig / (sig.std() + 1e-8)
    sw = s * np.hanning(len(sig))
    nfft = max(1024, 1 << int(np.ceil(np.log2(len(sig)))))
    spec = np.abs(np.fft.rfft(sw, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / FS)
    m = (freqs >= LO) & (freqs <= HI)
    if not m.any():
        return np.nan
    return freqs[m][np.argmax(spec[m])] * 60.0


def analyze(name, roi_path, gt_path):
    roi = np.load(roi_path, allow_pickle=False)
    gt = np.loadtxt(gt_path)
    n = min(len(roi), len(gt))
    bvp = gt[:n, 0]
    T = roi.shape[0]
    bh = roi.shape[1] // N_SUB
    sub_g = np.zeros((T, N_SUB * N_SUB), np.float32)
    for i in range(N_SUB):
        for j in range(N_SUB):
            idx = i * N_SUB + j
            sub_g[:, idx] = roi[:, i*bh:(i+1)*bh, j*bh:(j+1)*bh, 1].mean(axis=(1, 2))
    sos = butter_bandpass()
    whole, gts, subs = [], [], [[] for _ in range(N_SUB * N_SUB)]
    for s in range(0, n - SEQ + 1, STEP):
        e = s + SEQ
        t_hr = fft_hr(bvp[s:e])
        if np.isnan(t_hr):
            continue
        gts.append(t_hr)
        whole.append(fft_hr(roi[s:e].reshape(e - s, -1, 3)[:, :, 1].mean(axis=1)))
        for idx in range(N_SUB * N_SUB):
            seg = sub_g[s:e, idx]
            subs[idx].append(fft_hr(sosfiltfilt(sos, seg - seg.mean())))
    gts = np.array(gts, np.float32)
    w = np.array(whole, np.float32)
    mm = np.isfinite(w) & np.isfinite(gts)
    r_w = np.corrcoef(w[mm], gts[mm])[0, 1]
    mae_w = np.mean(np.abs(w[mm] - gts[mm]))
    best = []
    for idx in range(N_SUB * N_SUB):
        v = np.array(subs[idx], np.float32)
        m2 = np.isfinite(v) & np.isfinite(gts)
        vv, gg = v[m2], gts[m2]
        if len(vv) < 3:
            continue
        r = np.corrcoef(vv, gg)[0, 1]
        best.append((abs(r), r, np.mean(np.abs(vv - gg))))
    best.sort(reverse=True)
    print(f"[{name}] 窗口={len(gts)}")
    print(f"  整体G均: MAE={mae_w:.2f} r={r_w:+.3f}")
    print(f"  最优子块: r={best[0][1]:+.3f} MAE={best[0][2]:.2f} (top3: {[(round(r,3), round(m,2)) for _, r, m in best[:3]]})")
    return r_w, mae_w


def main():
    base = os.path.expanduser("~/rppg")
    a32 = analyze("32x32(旧tracker)", f"{base}/datasets/combined/subject51_a0/auto_face_roi_32x32.npy",
                  f"{base}/datasets/combined/subject51_a0/ground_truth.txt")
    a64 = analyze("64x64(新tracker)", f"{base}/datasets/MMPD_raw/test64/auto_face_roi_64x64.npy",
                  f"{base}/datasets/MMPD_raw/test64/ground_truth.txt")
    print(f"\n结论: 64x64 整体r {a64[0]:+.3f} vs 32x32 {a32[0]:+.3f}; "
          f"64x64 MAE {a64[1]:.2f} vs 32x32 {a32[1]:.2f}")


if __name__ == "__main__":
    main()
