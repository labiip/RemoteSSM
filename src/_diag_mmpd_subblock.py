"""MMPD face_roi 子块信息量验证.

face_roi (T,32,32,3) 分成 NxN 子块, 每块 G 通道平均 → 带通滤波 → FFT HR vs BVP HR.
回答: 视频局部是否存在可学的脉搏信息 (SOTA 证明存在, 但需像素级时空建模).
用法: python3 _diag_mmpd_subblock.py <folder_name>
"""
import os, sys, re
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = os.path.expanduser("~/rppg/datasets/combined")
FS = 30.0
SEQ = 300
STEP = 30
LO, HI = 0.7, 3.0
N_SUB = 4  # 4x4 子块


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


def main():
    folder = sys.argv[1]
    path = os.path.join(ROOT, folder)
    gt = np.loadtxt(os.path.join(path, "ground_truth.txt"))
    if gt.shape[1] == 3:
        bvp = gt[:, 0].flatten().astype(np.float32)
    else:
        bvp = gt[0].flatten().astype(np.float32)
    roi = np.load(os.path.join(path, "auto_face_roi_32x32.npy"), allow_pickle=False).astype(np.float32)
    n = min(len(roi), len(bvp))
    roi, bvp = roi[:n], bvp[:n]
    T = n
    sos = butter_bandpass()

    # 4x4 子块 G 平均
    sub_g = np.zeros((T, N_SUB * N_SUB), np.float32)
    bh = 32 // N_SUB
    for i in range(N_SUB):
        for j in range(N_SUB):
            idx = i * N_SUB + j
            sub_g[:, idx] = roi[:, i*bh:(i+1)*bh, j*bh:(j+1)*bh, 1].mean(axis=(1, 2))

    rows = {"sub": [[] for _ in range(N_SUB * N_SUB)], "whole": [], "gt": []}
    for s in range(0, T - SEQ + 1, STEP):
        e = s + SEQ
        t_hr = fft_hr(bvp[s:e])
        if np.isnan(t_hr):
            continue
        rows["gt"].append(t_hr)
        rows["whole"].append(fft_hr(roi[s:e, :, :, 1].mean(axis=(1, 2))))
        for idx in range(N_SUB * N_SUB):
            seg = sub_g[s:e, idx]
            segf = sosfiltfilt(sos, seg - seg.mean())
            rows["sub"][idx].append(fft_hr(segf))

    gt = np.array(rows["gt"], np.float32)
    print(f"[{folder}] T={T} 窗口={len(gt)}")
    # 整体
    w = np.array(rows["whole"], np.float32)
    mm = np.isfinite(w) & np.isfinite(gt)
    print(f"  face_roi整体G均: MAE={np.mean(np.abs(w[mm]-gt[mm])):.2f} r={np.corrcoef(w[mm], gt[mm])[0,1]:+.3f}")
    # 各子块
    best = []
    for idx in range(N_SUB * N_SUB):
        v = np.array(rows["sub"][idx], np.float32)
        m2 = np.isfinite(v) & np.isfinite(gt)
        vv, gg = v[m2], gt[m2]
        if len(vv) < 3:
            continue
        r = np.corrcoef(vv, gg)[0, 1]
        mae = np.mean(np.abs(vv - gg))
        best.append((abs(r), r, mae, idx))
    best.sort(reverse=True)
    for ar, r, mae, idx in best[:6]:
        print(f"  sub{idx:02d}: r={r:+.3f} MAE={mae:.2f}")
    if best:
        print(f"  最优子块: r={best[0][1]:+.3f} MAE={best[0][2]:.2f}")


if __name__ == "__main__":
    main()
