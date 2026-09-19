"""benchmark: preprocess_mmpd 管线单帧耗时 (p1_0 前 300 帧)."""
import os, sys, time
import numpy as np
import scipy.io
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_mmpd import RobustFaceTracker, MIN_FACE, DETECT_INTERVAL

mat = os.path.expanduser("~/rppg/datasets/MMPD_raw/p1_0.mat")
d = scipy.io.loadmat(mat, variable_names=["video"])
video = d["video"]
n = 300
tracker = RobustFaceTracker(min_size=MIN_FACE, detect_interval=DETECT_INTERVAL)
t0 = time.time()
det_hits = 0
for t in range(n):
    rgb = video[t]
    frame_bgr = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    res = tracker.process_frame(frame_bgr)
    roi_tuple = res[0] if res is not None else None
    if roi_tuple is not None:
        det_hits += 1
dt = time.time() - t0
print(f"300 frames: {dt:.1f}s => {dt/n*1000:.0f} ms/frame, det_hits={det_hits}/300")
