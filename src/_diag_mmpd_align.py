"""MMPD 时间对齐 + ROI 质量诊断.

验证两件事:
1. bgr G 信号与 BVP 波形在 ±60 帧 lag 范围内的最大 |r|:
   - 若有明确 lag 且 |r|>0.2 → 信号里有脉搏但时间错位
   - 若全≈0 → 信号本身无脉搏信息 (ROI 失败/背景)
2. face_roi 冻结程度: 相邻帧相同像素比例 (冻结=人脸检测失败)
用法: python3 _diag_mmpd_align.py 51 52
"""
import os, sys, re
import numpy as np

ROOT = os.path.expanduser("~/rppg/datasets/combined")
LAGS = range(-60, 61)


def load_folder(path):
    gt = np.loadtxt(os.path.join(path, "ground_truth.txt"))
    if gt.ndim != 2 or gt.shape[1] < 3:
        return None, None, None
    bvp = gt[:, 0].flatten().astype(np.float32)
    bgr = np.load(os.path.join(path, "auto_bgr_feat_5roi.npy"), allow_pickle=False).astype(np.float32)
    roi = np.load(os.path.join(path, "auto_face_roi_32x32.npy"), allow_pickle=False)
    return bgr, roi, bvp


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
    s_lo, s_hi = int(sys.argv[1]), int(sys.argv[2])
    folders = sorted(os.listdir(ROOT))
    for folder in folders:
        m = re.search(r"(\d+)", folder)
        if not m:
            continue
        sid = int(m.group(1))
        if not (s_lo <= sid <= s_hi):
            continue
        try:
            bgr, roi, bvp = load_folder(os.path.join(ROOT, folder))
        except Exception as e:
            print(f"[{folder}] load fail: {e}")
            continue
        if bgr is None:
            continue
        n = min(len(bgr), len(roi), len(bvp))
        g = bgr[:n, 1].astype(np.float64)
        g = (g - g.mean()) / (g.std() + 1e-8)
        bv = bvp[:n].astype(np.float64)
        bv = (bv - bv.mean()) / (bv.std() + 1e-8)
        # 时间相关性扫描
        lags, rs = [], []
        for lag in LAGS:
            r = corr_at_lag(g, bv, lag)
            lags.append(lag); rs.append(abs(r))
        best = int(lags[int(np.argmax(rs))])
        best_r = float(np.max(rs))
        # ROI 冻结程度: 相邻帧 L1 差异均值 (0=完全冻结)
        r0 = roi[:n].reshape(n, -1).astype(np.float32)
        diff = np.abs(np.diff(r0, axis=0)).mean()
        uniq_ratio = len(np.unique(r0[:2000])) / max(1, (2000 * r0.shape[1]))
        print(f"[{folder}] n={n} | bgrG_std={bgr[:n, 1].std():.2f} "
              f"| ROI帧间差={diff:.1f} | lag_best={best:+d} r={best_r:.3f} | "
              f"G-BVP r(lag0)={abs(corr_at_lag(g, bv, 0)):.3f}")


if __name__ == "__main__":
    main()
