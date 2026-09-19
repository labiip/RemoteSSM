"""MMPD 信号信息量诊断: 从 face_roi 与 bgr 直接 FFT HR, 对比 BVP-derived HR.

回答: MMPD 数据里心率信息到底在哪?
  - face_roi 平均像素序列的 FFT HR r 若显著 >0 → 信息在空间支路, 模型应能利用
  - bgr (CHROM/POS) 的 r 若≈0 → 信号支路在 MMPD 上无效 (与 baseline 一致)
  - 两者都≈0 → 数据/预处理问题 (ROI 错位/运动伪影/BVP 标签问题)
用法: python3 _diag_mmpd_signal.py 51 54
"""
import os, sys, re
import numpy as np

ROOT = os.path.expanduser("~/rppg/datasets/combined")
FS = 30.0
SEQ = 300
STEP = 30
LO, HI = 0.7, 3.0


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


def load_folder(path):
    gt = np.loadtxt(os.path.join(path, "ground_truth.txt"))
    if gt.ndim != 2:
        return None, None, None
    if gt.shape[0] == 3:
        bvp = gt[0].flatten().astype(np.float32)
    elif gt.shape[1] == 3:
        bvp = gt[:, 0].flatten().astype(np.float32)
    else:
        return None, None, None
    bgr = np.load(os.path.join(path, "auto_bgr_feat_5roi.npy"), allow_pickle=False).astype(np.float32)
    roi = np.load(os.path.join(path, "auto_face_roi_32x32.npy"), allow_pickle=False).astype(np.float32)
    return bgr, roi, bvp


def main():
    s_lo, s_hi = int(sys.argv[1]), int(sys.argv[2])
    rows = {"faceg": [], "faceall": [], "bgrg": [], "chrom": [], "gt": []}
    folders = sorted(os.listdir(ROOT))
    n_win = 0
    for folder in folders:
        m = re.search(r"(\d+)", folder)
        if not m:
            continue
        sid = int(m.group(1))
        if not (s_lo <= sid <= s_hi):
            continue
        try:
            bgr, roi, bvp = load_folder(os.path.join(ROOT, folder))
        except Exception:
            continue
        if bgr is None or roi is None or bvp is None:
            continue
        n = min(len(bgr), len(roi), len(bvp))
        if n < SEQ:
            continue
        bgr, roi, bvp = bgr[:n], roi[:n], bvp[:n]
        # face_roi: (N,32,32,3) BGR; 平均像素
        fr = roi.reshape(n, -1, 3)
        face_g = fr[:, :, 1].mean(axis=1)   # G 通道平均
        face_all = fr.mean(axis=(1, 2))     # 全通道平均
        for s in range(0, n - SEQ + 1, STEP):
            e = s + SEQ
            t_hr = fft_hr(bvp[s:e])
            if np.isnan(t_hr):
                continue
            rows["faceg"].append(fft_hr(face_g[s:e]))
            rows["faceall"].append(fft_hr(face_all[s:e]))
            rows["bgrg"].append(fft_hr(bgr[s:e, 1]))
            rows["gt"].append(t_hr)
            n_win += 1

    print(f"MMPD subjects {s_lo}-{s_hi} | 窗口数: {n_win}")
    print("=" * 70)
    print(f"{'信号':16s} {'MAE':>7s} {'r(gt)':>7s}")
    print("=" * 70)
    gt = np.array(rows["gt"], np.float32)
    for name, key in [("face_roi G均", "faceg"), ("face_roi 全均", "faceall"),
                      ("bgr G通道", "bgrg"), ("GT-BVP", "gt")]:
        v = np.array(rows[key], np.float32)
        mm = np.isfinite(v) & np.isfinite(gt)
        vv, gg = v[mm], gt[mm]
        if len(vv) == 0:
            print(f"{name:16s}    nan    nan")
            continue
        mae = np.mean(np.abs(vv - gg))
        r = np.corrcoef(vv, gg)[0, 1] if len(vv) > 2 else float("nan")
        print(f"{name:16s} {mae:7.2f} {r:+7.3f}")


if __name__ == "__main__":
    main()
