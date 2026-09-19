"""MMPD 改进版信号提取验证: 检测失败时用 MIL tracker 兜底, 避免信号冻结.

对比旧管线 (Haar 失败→冻结) 的信号质量.
输出: face 检测成功率 + bgr G 与 GT_ppg 的相关性 (lag 扫描).
用法: python3 _extract_mmpd_fix.py <mat_path>
"""
import os, sys
import numpy as np
import scipy.io
import cv2

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

    bgr_sig = np.zeros((T, 3), dtype=np.float32)
    tracker = None
    track_box = None
    n_det = 0
    n_track = 0
    last_fused = np.array([127.0, 127.0, 127.0], dtype=np.float32)

    for t in range(T):
        rgb = video[t]
        frame_bgr = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        h_img, w_img = frame_bgr.shape[:2]
        box = None

        # 每帧 Haar 检测 (低 min_size)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            box = [x, y, w, h]
            n_det += 1
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame_bgr, tuple(box))
        elif tracker is not None:
            ok, tb = tracker.update(frame_bgr)
            if ok:
                x, y, w, h = [int(v) for v in tb]
                x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
                w = max(20, min(w, w_img - x)); h = max(20, min(h, h_img - y))
                box = [x, y, w, h]
                n_track += 1

        if box is not None:
            sx, sy, sw, sh = box
            # 5-ROI 融合 (同旧管线)
            rois = [
                (sx + int(sw * 0.30), sy + int(sh * 0.05), sx + int(sw * 0.70), sy + int(sh * 0.25)),
                (sx + int(sw * 0.10), sy + int(sh * 0.05), sx + int(sw * 0.30), sy + int(sh * 0.25)),
                (sx + int(sw * 0.70), sy + int(sh * 0.05), sx + int(sw * 0.90), sy + int(sh * 0.25)),
                (sx + int(sw * 0.10), sy + int(sh * 0.45), sx + int(sw * 0.40), sy + int(sh * 0.70)),
                (sx + int(sw * 0.60), sy + int(sh * 0.45), sx + int(sw * 0.90), sy + int(sh * 0.70)),
            ]
            means = []
            for (x1, y1, x2, y2) in rois:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 <= x1 or y2 <= y1:
                    means.append(last_fused)
                else:
                    m = frame_bgr[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
                    means.append(m)
            fused = (0.4 * np.array(means[0]) + 0.1 * np.array(means[1]) + 0.1 * np.array(means[2])
                     + 0.2 * np.array(means[3]) + 0.2 * np.array(means[4]))
            last_fused = fused
        # 检测+追踪都失败: 保持上次值 (冻结, 但应更少发生)

        bgr_sig[t] = last_fused

    det_rate = n_det / T
    track_rate = n_track / T
    g = bgr_sig[:, 1].astype(np.float64)
    gz = (g - g.mean()) / (g.std() + 1e-8)
    ppgz = (ppg.astype(np.float64) - ppg.mean()) / (ppg.std() + 1e-8)
    rs = [abs(corr_at_lag(gz, ppgz, lag)) for lag in LAGS]
    bi = int(np.argmax(rs))
    print(f"{os.path.basename(path)}: T={T} | Haar成功={det_rate:.1%} 追踪={track_rate:.1%} "
          f"失败冻结={1-det_rate-track_rate:.1%} | bgrG_std={g.std():.2f} | "
          f"G-BVP lag_best={LAGS[bi]:+d} r={rs[bi]:.3f} | r(lag0)={abs(corr_at_lag(gz, ppgz, 0)):.3f}")


if __name__ == "__main__":
    main()
