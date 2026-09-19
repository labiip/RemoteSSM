"""MMPD 预处理: .mat → UBFC 三件套.

MMPD .mat 结构 (scipy.io.loadmat 可读, v5 格式):
  - video:   (T, H=320, W=240, C=3) float32 [0,1] RGB
  - GT_ppg:  (1, T) float64 PPG 波形
  - 其余为元数据 (light/motion/exercise/...)

输出目录结构 (combined):
  subject{global_id}_a{activity}/
    auto_bgr_feat_5roi.npy   (T, 3) float32  BGR 5-ROI 融合信号
    auto_face_roi_32x32.npy  (T, 32, 32, 3) uint8
    ground_truth.txt         (T, 3) [BVP, HR, timestamp]

MMPD subject1..33 → global id 51..83 (避开 UBFC 的 1..49)
"""
import os
import re
import glob
import numpy as np
import scipy.io
import cv2
from scipy.signal import find_peaks

from face_tracker import RobustFaceTracker

FS = 30.0
ROI_SIZE = 64          # 64x64: 保留更多空间细节 (32x32 已验证是信息瓶颈)
SRC = "~/rppg/datasets/MMPD"
DST = "~/rppg/datasets/combined"
SUBJECT_OFFSET = 50
MIN_FACE = (60, 60)    # 全量 660 视频实测检测率 99.9%; 更小阈值使 Haar 多扫 ~1.5x 尺度, 收益微小
DETECT_INTERVAL = 1    # 每帧检测 (运动场景下逐帧检测, 避免错位冻结)


def peak_detection_hr(ppg, fs=FS):
    """从 PPG 波形峰值检测得到逐帧 HR (60/IBI 线性插值)."""
    T = len(ppg)
    sig = np.asarray(ppg, dtype=np.float64)
    sig = sig - np.mean(sig)
    sig = sig / (np.std(sig) + 1e-8)
    peaks, _ = find_peaks(sig, distance=int(0.4 * fs))  # 0.4s -> <=150bpm
    if len(peaks) < 2:
        return np.full(T, 0.0, dtype=np.float32)
    hr_positions = peaks[1:].astype(np.float64)
    hr_at_peak = 60.0 * fs / np.diff(peaks).astype(np.float64)
    hr_full = np.interp(np.arange(T, dtype=np.float64), hr_positions, hr_at_peak)
    return np.clip(hr_full, 40.0, 180.0).astype(np.float32)


def process_mat(mat_path, out_dir):
    d = scipy.io.loadmat(mat_path, variable_names=["video", "GT_ppg"])
    video = d["video"]                      # (T,H,W,C) float32 [0,1] RGB
    ppg = np.asarray(d["GT_ppg"], dtype=np.float32).reshape(-1)
    T = min(video.shape[0], len(ppg))
    video = video[:T]
    ppg = ppg[:T]

    tracker = RobustFaceTracker(min_size=MIN_FACE, detect_interval=DETECT_INTERVAL)
    bgr_signal = np.zeros((T, 3), dtype=np.float32)
    face_rois = np.zeros((T, ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
    last_fused = np.array([127.0, 127.0, 127.0], dtype=np.float32)
    last_roi = np.full((ROI_SIZE, ROI_SIZE, 3), 128, dtype=np.uint8)
    mil_tracker = None
    last_face_box = None

    for t in range(T):
        rgb = video[t]                      # (H,W,C) float [0,1]
        frame_bgr = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        h_img, w_img = frame_bgr.shape[:2]
        res = tracker.process_frame(frame_bgr)
        roi_tuple = res[0] if res is not None else None
        face_box = tracker.get_face_box() if roi_tuple is not None else None

        if roi_tuple is not None:
            # Haar 成功 (MMPD 实测 ~100%): 权威框, 丢弃 MIL
            fh_c, fh_l, fh_r, ch_l, ch_r = roi_tuple
            fused = (
                0.4 * np.array(fh_c[:3], dtype=np.float32)
                + 0.1 * np.array(fh_l[:3], dtype=np.float32)
                + 0.1 * np.array(fh_r[:3], dtype=np.float32)
                + 0.2 * np.array(ch_l[:3], dtype=np.float32)
                + 0.2 * np.array(ch_r[:3], dtype=np.float32)
            )
            last_fused = fused
            mil_tracker = None
            if face_box is not None:
                last_face_box = face_box
        elif last_face_box is not None:
            # Haar 失败 → MIL 续帧 (仅在首个失败帧初始化, 避免每帧重建的开销)
            if mil_tracker is None:
                mil_tracker = cv2.TrackerMIL_create()
                mil_tracker.init(frame_bgr, tuple(last_face_box))
            ok, tb = mil_tracker.update(frame_bgr)
            if ok:
                x, y, w, h = [int(v) for v in tb]
                x = max(0, min(x, w_img - 1)); y = max(0, min(y, h_img - 1))
                w = max(20, min(w, w_img - x)); h = max(20, min(h, h_img - y))
                face_box = (x, y, w, h)
                last_face_box = face_box

        if face_box is not None:
            fx, fy, fw, fh = face_box
            fx, fy = max(0, fx), max(0, fy)
            crop = frame_bgr[fy:fy + fh, fx:fx + fw]
            if crop.size > 0:
                last_roi = cv2.resize(crop, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR)

        bgr_signal[t] = last_fused
        face_rois[t] = last_roi

    hr = peak_detection_hr(ppg)
    timestamp = np.arange(T, dtype=np.float32) / FS

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "auto_bgr_feat_5roi.npy"), bgr_signal)
    np.save(os.path.join(out_dir, f"auto_face_roi_{ROI_SIZE}x{ROI_SIZE}.npy"), face_rois)
    np.savetxt(os.path.join(out_dir, "ground_truth.txt"),
               np.stack([ppg, hr, timestamp], axis=1), fmt="%.6f")
    return T


def main():
    import sys
    mat_files = sorted(glob.glob(os.path.join(SRC, "subject*", "*.mat")))
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mat_files = [f for i, f in enumerate(mat_files) if i % num_workers == worker_id]
    print(f"[worker {worker_id}/{num_workers}] 处理 {len(mat_files)} 个 .mat 文件", flush=True)
    done = skipped = 0
    for mat_path in mat_files:
        dname = os.path.basename(os.path.dirname(mat_path))
        sid = int(re.search(r"(\d+)", dname).group(1))
        global_id = sid + SUBJECT_OFFSET
        fname = os.path.basename(mat_path)
        act = re.search(r"_(\d+)\.mat$", fname).group(1)
        out_dir = os.path.join(DST, f"subject{global_id}_a{act}")
        out_roi = os.path.join(out_dir, f"auto_face_roi_{ROI_SIZE}x{ROI_SIZE}.npy")
        if os.path.exists(out_roi):
            skipped += 1
            continue
        try:
            T = process_mat(mat_path, out_dir)
            done += 1
            print(f"[{done}] {dname}/{fname} -> subject{global_id}_a{act} ({T}帧)", flush=True)
        except Exception as e:
            print(f"[FAIL] {mat_path}: {e}", flush=True)
    print(f"完成 {done}, 跳过 {skipped}, 总计 {len(mat_files)}", flush=True)


if __name__ == "__main__":
    main()
