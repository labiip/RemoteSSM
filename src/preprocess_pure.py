"""PURE 预处理: zip → UBFC 三件套.

PURE zip 结构:
  <subj>-<task>/Image<timestamp_ns>.png  (2026 张 PNG, 30fps)
  <subj>-<task>.json:
    /FullPackage: [{Timestamp, Value:{waveform, pulseRate, ...}}, ...]  (PPG 50Hz)
    /Image:       [{Timestamp, FrameID:'/CameraFrame'}, ...]

pulseRate = 血氧仪官方 HR, waveform = PPG 波形, 二者重采样到帧时间戳.

PURE subject1..10 → global id 101..110
"""
import os
import re
import glob
import json
import zipfile
import numpy as np
import cv2

from face_tracker import RobustFaceTracker

FS = 30.0
ROI_SIZE = 32
SRC = "~/rppg/datasets/PURE"
DST = "~/rppg/datasets/combined"
SUBJECT_OFFSET = 100
MIN_FACE = (80, 80)   # PURE 视频 640x480


def process_zip(zip_path, out_dir):
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()

    pngs = []
    for n in names:
        m = re.search(r"Image(\d+)\.png$", n)
        if m:
            pngs.append((int(m.group(1)), n))
    pngs.sort(key=lambda x: x[0])
    if not pngs:
        raise RuntimeError("zip 内无 PNG 帧")

    json_name = [n for n in names if n.endswith(".json")][0]
    d = json.loads(z.read(json_name).decode())
    fp = d["/FullPackage"]

    fp_ts = np.array([e["Timestamp"] for e in fp], dtype=np.float64)
    waveform = np.array([e["Value"]["waveform"] for e in fp], dtype=np.float32)
    pulse_rate = np.array([e["Value"]["pulseRate"] for e in fp], dtype=np.float32)

    frame_ts = np.array([ts for ts, _ in pngs], dtype=np.float64)
    bvp = np.interp(frame_ts, fp_ts, waveform).astype(np.float32)
    hr = np.interp(frame_ts, fp_ts, pulse_rate).astype(np.float32)

    T = len(pngs)
    tracker = RobustFaceTracker(min_size=MIN_FACE, detect_interval=2)
    bgr_signal = np.zeros((T, 3), dtype=np.float32)
    face_rois = np.zeros((T, ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
    last_fused = np.array([127.0, 127.0, 127.0], dtype=np.float32)
    last_roi = np.full((ROI_SIZE, ROI_SIZE, 3), 128, dtype=np.uint8)

    for i, (ts, name) in enumerate(pngs):
        data = z.read(name)
        frame_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            bgr_signal[i] = last_fused
            face_rois[i] = last_roi
            continue

        res = tracker.process_frame(frame_bgr)
        roi_tuple = res[0] if res is not None else None
        face_box = tracker.get_face_box() if roi_tuple is not None else None

        if roi_tuple is not None:
            fh_c, fh_l, fh_r, ch_l, ch_r = roi_tuple
            fused = (
                0.4 * np.array(fh_c[:3], dtype=np.float32)
                + 0.1 * np.array(fh_l[:3], dtype=np.float32)
                + 0.1 * np.array(fh_r[:3], dtype=np.float32)
                + 0.2 * np.array(ch_l[:3], dtype=np.float32)
                + 0.2 * np.array(ch_r[:3], dtype=np.float32)
            )
            last_fused = fused

        if face_box is not None:
            fx, fy, fw, fh = face_box
            fx, fy = max(0, fx), max(0, fy)
            crop = frame_bgr[fy:fy + fh, fx:fx + fw]
            if crop.size > 0:
                last_roi = cv2.resize(crop, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR)

        bgr_signal[i] = last_fused
        face_rois[i] = last_roi

    timestamp = (frame_ts - frame_ts[0]) / 1e9  # 秒

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "auto_bgr_feat_5roi.npy"), bgr_signal)
    np.save(os.path.join(out_dir, f"auto_face_roi_{ROI_SIZE}x{ROI_SIZE}.npy"), face_rois)
    np.savetxt(os.path.join(out_dir, "ground_truth.txt"),
               np.stack([bvp, hr, timestamp], axis=1), fmt="%.6f")
    return T


def main():
    import sys
    zips = sorted(glob.glob(os.path.join(SRC, "*.zip")))
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    zips = [f for i, f in enumerate(zips) if i % num_workers == worker_id]
    print(f"[worker {worker_id}/{num_workers}] 处理 {len(zips)} 个 zip", flush=True)
    done = skipped = 0
    for zp in zips:
        name = os.path.basename(zp)
        m = re.search(r"(\d+)-(\d+)\.zip$", name)
        if not m:
            continue
        subj = int(m.group(1))
        task = int(m.group(2))
        global_id = subj + SUBJECT_OFFSET
        out_dir = os.path.join(DST, f"subject{global_id}_a{task}")
        if os.path.exists(os.path.join(out_dir, "ground_truth.txt")):
            skipped += 1
            continue
        try:
            T = process_zip(zp, out_dir)
            done += 1
            print(f"[{done}] {name} -> subject{global_id}_a{task} ({T}帧)", flush=True)
        except Exception as e:
            print(f"[FAIL] {name}: {e}", flush=True)
    print(f"完成 {done}, 跳过 {skipped}, 总计 {len(zips)}", flush=True)


if __name__ == "__main__":
    main()
