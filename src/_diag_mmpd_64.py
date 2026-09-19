"""MMPD 64x64 face_roi 信息量验证 (对比 32x32).

从原始 .mat 用改进 tracker 提取 64x64 ROI, 子块带通滤波 FFT HR vs GT_ppg.
验证: 分辨率提升是否能显著提高可学信息上限.
用法: python3 _diag_mmpd_64.py <mat_path>
"""
import os, sys
import numpy as np
import scipy.io
import cv2
from scipy.signal import butter, sosfiltfilt

FS = 30.0
SEQ = 300
STEP = 30
LO, HI = 0.7, 3.0
N_SUB = 8  # 8x8 子块


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
    path = sys.argv[1]
    d = scipy.io.loadmat(path, variable_names=["video", "GT_ppg"])
    video = d["video"]
    ppg = np.asarray(d["GT_ppg"], dtype=np.float32).reshape(-1)
    T = min(video.shape[0], len(ppg))
    video = video[:T]
    ppg = ppg[:T]

    cascade = cv2.CascadeClassifier(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_alt2.xml"))
    if cascade.empty():
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    rois64 = np.zeros((T, 64, 64, 3), dtype=np.uint8)
    tracker = None
    last_roi = np.full((64, 64, 3), 128, dtype=np.uint8)
    for t in range(T):
        rgb = video[t]
        frame_bgr = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        h_img, w_img = frame_bgr.shape[:2]
        box = None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            box = [x, y, w, h]
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame_bgr, tuple(box))
        elif tracker is not None:
            ok, tb = tracker.update(frame_bgr)
            if ok:
                x, y, w, h = [int(v) for v in tb]
                x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
                w = max(20, min(w, w_img - x)); h = max(20, min(h, h_img - y))
                box = [x, y, w, h]
        if box is not None:
            sx, sy, sw, sh = box
            crop = frame_bgr[sy:sy + sh, sx:sx + sw]
            if crop.size > 0:
                last_roi = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
        rois64[t] = last_roi

    # 8x8 子块 G 平均 + 带通 + FFT HR
    sub_g = np.zeros((T, N_SUB * N_SUB), np.float32)
    bh = 64 // N_SUB
    for i in range(N_SUB):
        for j in range(N_SUB):
            idx = i * N_SUB + j
            sub_g[:, idx] = rois64[:, i*bh:(i+1)*bh, j*bh:(j+1)*bh, 1].mean(axis=(1, 2))
    sos = butter_bandpass()
    rows = {"whole": [], "gt": []}
    rows_sub = [[] for _ in range(N_SUB * N_SUB)]
    for s in range(0, T - SEQ + 1, STEP):
        e = s + SEQ
        t_hr = fft_hr(ppg[s:e])
        if np.isnan(t_hr):
            continue
        rows["gt"].append(t_hr)
        rows["whole"].append(fft_hr(rois64[s:e].reshape(e - s, -1, 3)[:, :, 1].mean(axis=1)))
        for idx in range(N_SUB * N_SUB):
            seg = sub_g[s:e, idx]
            rows_sub[idx].append(fft_hr(sosfiltfilt(sos, seg - seg.mean())))

    gt = np.array(rows["gt"], np.float32)
    w = np.array(rows["whole"], np.float32)
    mm = np.isfinite(w) & np.isfinite(gt)
    print(f"[{path.split('/')[-1]}] 64x64 ROI T={T} 窗口={len(gt)}")
    print(f"  整体G均: MAE={np.mean(np.abs(w[mm]-gt[mm])):.2f} r={np.corrcoef(w[mm], gt[mm])[0,1]:+.3f}")
    best = []
    for idx in range(N_SUB * N_SUB):
        v = np.array(rows_sub[idx], np.float32)
        m2 = np.isfinite(v) & np.isfinite(gt)
        vv, gg = v[m2], gt[m2]
        if len(vv) < 3:
            continue
        r = np.corrcoef(vv, gg)[0, 1]
        mae = np.mean(np.abs(vv - gg))
        best.append((abs(r), r, mae, idx))
    best.sort(reverse=True)
    print(f"  最优子块(8x8): r={best[0][1]:+.3f} MAE={best[0][2]:.2f}")
    for ar, r, mae, idx in best[:3]:
        print(f"    sub{idx:02d}: r={r:+.3f} MAE={mae:.2f}")


if __name__ == "__main__":
    main()
