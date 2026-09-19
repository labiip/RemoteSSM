"""通用 step30 评测: 任意 MMPD 64px checkpoint 在 test 78-83 的数值 (补 4.7 同口径 2x2).

用法:
  python _eval_ckpt_step30.py <ckpt> face     # face-only: forward 时信号输入置零
  python _eval_ckpt_step30.py <ckpt> dual     # dual-path: 正常输入

评测口径与 evaluate 一致: step30, 0.7-3.0 Hz, FFT(BVP) soft-argmax, BVP-HR 标签.
"""
import os
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
config.TRAIN_CONFIG["VAL_STEP"] = 30

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from dataset import TitanrPPGDataset
from model import RemoteSSM
from utils import DifferentiablePhysioExtractor, safe_torch_load


def evaluate_ckpt(model_path, mode="dual", test_subjects=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if test_subjects is None:
        test_subjects = list(range(78, 84))
    seq_len = int(COMMON_CONFIG["SEQ_LEN"])
    fs = float(COMMON_CONFIG["FS"])
    roi_size = int(TRAIN_CONFIG["FACE_ROI_SIZE"])

    model = RemoteSSM(d_model=COMMON_CONFIG["D_MODEL"], seq_len=seq_len,
                      n_ssm_blocks=COMMON_CONFIG.get("N_SSM_BLOCKS", 2),
                      roi_size=roi_size, dropout=0.15, fs=fs, use_fdf=True).to(device)
    ck = safe_torch_load(model_path, map_location=device)
    sd = ck.get("model_state_dict", ck)
    m, u = model.load_state_dict(sd, strict=False)
    if m:
        print(f"[WARN] missing {len(m)} keys")
    model.eval()

    ds = TitanrPPGDataset(root_dir=TRAIN_CONFIG["DATASET_ROOT"], seq_len=seq_len,
                          step=30, subject_ids=test_subjects, split_name="test")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[INFO] {mode} | {os.path.basename(model_path)} | {len(ds)} samples | {test_subjects}")

    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            face = batch["face_roi"].to(device)
            xr = batch["input_real"].to(device)
            xi = batch["input_imag"].to(device)
            if mode == "face":      # face-only: 与训练一致, 信号输入置零
                xr = torch.zeros_like(xr)
                xi = torch.zeros_like(xi)
            thr = batch["target_hr"].cpu().numpy()
            bvp_p, _ = model(face_roi=face, x_real=xr, x_imag=xi, mode="finetune")
            hp = extractor(torch.clamp(bvp_p, -3.0, 3.0),
                           freq_range=PHYSIO_CONFIG["HR_BAND"],
                           temperature=0.05).cpu().numpy()
            all_p.extend(hp.flatten().tolist())
            all_t.extend(thr.flatten().tolist())

    p = np.array(all_p, np.float32)
    t = np.array(all_t, np.float32)
    msk = (np.isfinite(p) & np.isfinite(t) & (t >= 30) & (t <= 200) & (p >= 30) & (p <= 200))
    pp, tt = p[msk], t[msk]
    mae = float(np.mean(np.abs(pp - tt)))
    rmse = float(np.sqrt(np.mean((pp - tt) ** 2)))
    pear = float(np.corrcoef(pp, tt)[0, 1]) if len(pp) > 1 else 0.0
    print(f"\n[{mode}] {os.path.basename(model_path)}")
    print(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  Pearson: {pear:.4f}")
    print(f"  Acc<3: {np.mean(np.abs(pp - tt) <= 3) * 100:.1f}%  "
          f"Acc<5: {np.mean(np.abs(pp - tt) <= 5) * 100:.1f}%  "
          f"Acc<8: {np.mean(np.abs(pp - tt) <= 8) * 100:.1f}%")
    print(f"  Valid: {len(pp)}/{len(p)}")
    return mae, rmse, pear


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else ""
    mode = sys.argv[2] if len(sys.argv) > 2 else "dual"
    if not mp or not os.path.exists(mp):
        print(f"[ERROR] checkpoint not found: {mp}")
        sys.exit(1)
    evaluate_ckpt(mp, mode=mode)
