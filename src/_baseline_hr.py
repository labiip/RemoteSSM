"""经典基线: GREEN/CHROM/POS/ICA 直接 FFT HR vs BVP-FFT HR (同训练评估口径).

口径: seq_len=300, step=30, HR band 0.7-3.0 Hz, BVP-derived label.
用法: python3 _baseline_hr.py <lo> <hi> <name>
  例如: python3 _baseline_hr.py 40 50 UBFC-test
        python3 _baseline_hr.py 109 111 PURE-test
        python3 _baseline_hr.py 78 84 MMPD-test
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


def chrom_signal(bgr):
    R = bgr[:, 2].astype(np.float32); G = bgr[:, 1].astype(np.float32); B = bgr[:, 0].astype(np.float32)
    X = R - G
    Y = 0.5 * R + 0.5 * G - B
    X = X - X.mean(); Y = Y - Y.mean()
    a = X.std() / (Y.std() + 1e-8)
    return X - a * Y


def pos_signal(bgr):
    C = bgr.astype(np.float32) - bgr.astype(np.float32).mean(axis=0)
    P = np.array([[0, 1, -1], [-2, 1, 1]], np.float32)
    S = C @ P.T
    a = S[:, 0].std() / (S[:, 1].std() + 1e-8)
    return S[:, 0] + a * S[:, 1]


def ica_signal(bgr):
    try:
        from sklearn.decomposition import FastICA
    except Exception:
        return None
    C = bgr.astype(np.float32)
    C = C - C.mean(axis=0)
    try:
        ica = FastICA(n_components=3, max_iter=500, random_state=0)
        S = ica.fit_transform(C)
        # 选与 G 通道相关性最高的独立分量 (rPPG 文献惯例)
        g = bgr[:, 1].astype(np.float32)
        corrs = [abs(np.corrcoef(S[:, k], g)[0, 1]) for k in range(3)]
        return S[:, int(np.argmax(corrs))]
    except Exception:
        return None


def load_folder(path):
    gt = np.loadtxt(os.path.join(path, "ground_truth.txt"))
    if gt.ndim != 2:
        return None, None
    if gt.shape[0] == 3:
        bvp = gt[0].flatten().astype(np.float32)
    elif gt.shape[1] == 3:
        bvp = gt[:, 0].flatten().astype(np.float32)
    else:
        return None, None
    bgr = np.load(os.path.join(path, "auto_bgr_feat_5roi.npy"), allow_pickle=False).astype(np.float32)
    return bgr, bvp


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
            bgr, bvp = load_folder(os.path.join(ROOT, folder))
        except Exception:
            continue
        if bgr is None or bvp is None:
            continue
        n = min(len(bgr), len(bvp))
        if n < SEQ:
            continue
        bgr, bvp = bgr[:n], bvp[:n]
        for s in range(0, n - SEQ + 1, STEP):
            e = s + SEQ
            seg = bgr[s:e]
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

    print(f"{name} subjects {s_lo}-{s_hi} | 窗口数: {n_win}")
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
