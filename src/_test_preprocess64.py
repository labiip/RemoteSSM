"""64x64 预处理验证: 用已拷贝的 p1_0.mat 跑通 process_mat, 检查输出 + 信息量.

用法: python3 _test_preprocess64.py
"""
import os, sys
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from preprocess_mmpd import process_mat

OUT = os.path.expanduser("~/rppg/datasets/MMPD_raw/test64")


def fft_hr(sig):
    fs = 30.0
    sig = np.asarray(sig, np.float32)
    sig = sig - sig.mean()
    s = sig / (sig.std() + 1e-8)
    sw = s * np.hanning(len(sig))
    nfft = max(1024, 1 << int(np.ceil(np.log2(len(sig)))))
    spec = np.abs(np.fft.rfft(sw, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    m = (freqs >= 0.7) & (freqs <= 3.0)
    return freqs[m][np.argmax(spec[m])] * 60.0


def main():
    mat = os.path.expanduser("~/rppg/datasets/MMPD_raw/p1_0.mat")
    T = process_mat(mat, OUT)
    print(f"processed T={T}")
    roi = np.load(os.path.join(OUT, "auto_face_roi_64x64.npy"))
    bgr = np.load(os.path.join(OUT, "auto_bgr_feat_5roi.npy"))
    gt = np.loadtxt(os.path.join(OUT, "ground_truth.txt"))
    print(f"roi: {roi.shape} {roi.dtype} | bgr: {bgr.shape} | gt: {gt.shape}")
    # 整体 G 均 vs GT BVP HR
    bvp = gt[:, 0]
    n = min(len(roi), len(bvp))
    g = roi[:n].astype(np.float32).reshape(n, -1, 3)[:, :, 1].mean(axis=1)
    preds, gts = [], []
    for s in range(0, n - 300 + 1, 300):
        e = s + 300
        t_hr = fft_hr(bvp[s:e])
        if not np.isnan(t_hr):
            preds.append(fft_hr(g[s:e]))
            gts.append(t_hr)
    preds = np.array(preds); gts = np.array(gts)
    print(f"整体G均: MAE={np.mean(np.abs(preds-gts)):.2f} r={np.corrcoef(preds, gts)[0,1]:+.3f} (p1_0)")


if __name__ == "__main__":
    main()
