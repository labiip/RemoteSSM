"""标准全脸 ROI 基线: 用 auto_face_roi 均值 RGB 计算 GREEN/CHROM/POS/ICA.

文献标准做法 (整脸 ROI 均值信号), 替代 5-ROI 融合信号, 公平对比.
口径: seq_len=300, step=30, HR band 0.7-3.0 Hz, BVP-derived label.
用法: python3 _baseline_face_hr.py <lo> <hi> <name>
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


def chrom_signal(rgb):
    R = rgb[:, 2].astype(np.float32); G = rgb[:, 1].astype(np.float32); B = rgb[:, 0].astype(np.float32)
    X = R - G
    Y = 0.5 * R + 0.5 * G - B
    X = X - X.mean(); Y = Y - Y.mean()
    a = X.std() / (Y.std() + 1e-8)
    return X - a * Y


def pos_signal(rgb):
    C = rgb.astype(np.float32) - rgb.astype(np.float32).mean(axis=0)
    P = np.array([[0, 1, -1], [-2, 1, 1]], np.float32)
    S = C @ P.T
    a = S[:, 0].std() / (S[:, 1].std() + 1e-8)
    return S[:, 0] + a * S[:, 1]


def ica_signal(rgb):
    try:
        from sklearn.decomposition import FastICA
    except Exception:
        return None
    C = rgb.astype(np.float32)
    C = C - C.mean(axis=0)
    try:
        ica = FastICA(n_components=3, max_iter=500, random_state=0)
        S = ica.fit_transform(C)
        g = rgb[:, 1].astype(np.float32)
        corrs = [abs(np.corrcoef(S[:, k], g)[0, 1]) for k in range(3)]
        return S[:, int(np.argmax(corrs))]
    except Exception:
        return None


def load_face_signal(path):
    """从 face_roi 数组计算逐帧均值 RGB (BGR 存储顺序)."""
    gt = np.loadtxt(os.path.join(path, "ground_truth.txt"))
    if gt.ndim != 2 or gt.shape[1] != 3:
        return None, None
    bvp = gt[:, 0].flatten().astype(np.float32)
    for sz in (64, 32):
        roi_path = os.path.join(path, f"auto_face_roi_{sz}x{sz}.npy")
        if os.path.exists(roi_path):
            roi = np.load(roi_path, allow_pickle=False)      # (T,H,W,3) uint8 BGR
            rgb = roi.reshape(roi.shape[0], -1, 3).mean(axis=1).astype(np.float32)  # (T,3) BGR均值
            return rgb, bvp
    return None, None


def main():
    s_lo, s_hi, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rows = {"g": [], "chrom": [], "pos": [], "ica": [], "gt": []}
    n_win = 0
    for folder in sorted(os.listdir(ROOT)):
        m = re.search(r"(\d+)", folder)
        if not m:
            continue
        sid = int(m.group(1))
        if not (s_lo <= sid <= s_hi):
            continue
        try:
            rgb, bvp = load_face_signal(os.path.join(ROOT, folder))
        except Exception:
            continue
        if rgb is None or bvp is None:
            continue
        n = min(len(rgb), len(bvp))
        if n < SEQ:
            continue
        rgb, bvp = rgb[:n], bvp[:n]
        for s in range(0, n - SEQ + 1, STEP):
            e = s + SEQ
            seg = rgb[s:e]
            t_hr = fft_hr(bvp[s:e])
            if np.isnan(t_hr):
                continue
            rows["g"].append(fft_hr(seg[:, 1]))
            rows["chrom"].append(fft_hr(chrom_signal(seg)))
            rows["pos"].append(fft_hr(pos_signal(seg)))
            ic = ica_signal(seg)
            rows["ica"].append(fft_hr(ic) if ic is not None else np.nan)
            rows["gt"].append(t_hr)
            n_win += 1

    print(f"[全脸ROI基线] {name} subjects {s_lo}-{s_hi} | 窗口数: {n_win}")
    print("=" * 66)
    print(f"{'方法':12s} {'MAE':>7s} {'RMSE':>7s} {'r(gt)':>7s} {'Acc<5':>7s}")
    print("=" * 66)
    gt = np.array(rows["gt"], np.float32)
    for label, key in [("GREEN", "g"), ("CHROM", "chrom"), ("POS", "pos"), ("ICA", "ica")]:
        v = np.array(rows[key], np.float32)
        m = np.isfinite(v) & np.isfinite(gt)
        vv, gg = v[m], gt[m]
        mae = np.mean(np.abs(vv - gg))
        rmse = np.sqrt(np.mean((vv - gg) ** 2))
        r = np.corrcoef(vv, gg)[0, 1] if len(vv) > 2 else float("nan")
        a5 = np.mean(np.abs(vv - gg) <= 5) * 100 if len(vv) > 0 else 0
        print(f"{label:12s} {mae:7.2f} {rmse:7.2f} {r:+7.3f} {a5:6.1f}%")


if __name__ == "__main__":
    main()
