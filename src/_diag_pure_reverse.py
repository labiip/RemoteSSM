"""??? PURE test ??64x64 cache ?????????."""
import os, glob
import numpy as np

ROOT = os.path.expanduser("~/rppg/datasets/combined")

print("=== 1. PURE subject109/110 64x64 ROI ??? ===")
for f in sorted(glob.glob(os.path.join(ROOT, "subject109*/auto_face_roi_64x64.npy")) +
                glob.glob(os.path.join(ROOT, "subject110*/auto_face_roi_64x64.npy"))):
    a = np.load(f, allow_pickle=False)
    # ????????????: ?????? 128 ??(?????last_valid)
    uniq = np.unique(a.reshape(-1, 3).mean(axis=1))
    print(os.path.basename(os.path.dirname(f)), a.shape, "std=%.2f" % a.std(), "min=%.0f max=%.0f" % (a.min(), a.max()))

print("\n=== 2. ??? UBFC subject40 64x64 ROI ===")
f40 = glob.glob(os.path.join(ROOT, "subject40/auto_face_roi_64x64.npy"))
if f40:
    a = np.load(f40[0], allow_pickle=False)
    print(os.path.basename(os.path.dirname(f40[0])), a.shape, "std=%.2f" % a.std(), "min=%.0f max=%.0f" % (a.min(), a.max()))

print("\n=== 3. PURE cache ????????? ===")
caches = sorted(glob.glob(os.path.join(os.path.expanduser("~/rppg/rppg_project/cache"),
                                       "offline_cache_dir_cardio_v6_roi64_real_test_*seq300_step30")))
for c in caches:
    files = sorted(glob.glob(os.path.join(c, "*.pt")))
    hrs, subs = [], []
    for f in files[:2000]:
        try:
            import torch
            s = torch.load(f, map_location="cpu", weights_only=False)
            hrs.append(float(s["target_hr"])); subs.append(int(s.get("meta_subject", -1)))
        except Exception as e:
            print("load fail", os.path.basename(f), e)
    if hrs:
        hrs = np.array(hrs)
        print(os.path.basename(c), "n=%d" % len(files))
        print("  subjects:", sorted(set(subs)))
        print("  target_hr: min=%.1f max=%.1f mean=%.1f std=%.1f" % (hrs.min(), hrs.max(), hrs.mean(), hrs.std()))

