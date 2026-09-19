"""导出画图数据: 逐窗口 (预测HR, 参考HR, 时间) → npz.

用于论文 Fig: HR 轨迹图 + Bland-Altman 图.
用法: CUDA_VISIBLE_DEVICES=1 python3 _dump_plot_data.py
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
from model import RemoteSSM
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor, safe_torch_load

device = torch.device("cuda")
OUT = os.path.join(PROJ, "plot_data")
os.makedirs(OUT, exist_ok=True)

# (name, model_path, roi_size, cache_version, subjects, step)
# 三个面板都必须用论文表格 (Table 1) 主行对应的 checkpoint:
#   UBFC / PURE -> 联合 UBFC+PURE 训练 (0.87 / 0.31)
#   MMPD        -> mmpd_dual_pt_v30, val_step=30 选点 (13.69)
JOBS = [
    ("ubfc", "outputs/ubfc_pure/remotessm_ubfc_pure_best.pth", 32, "cardio_v5_bvphr",
     list(range(40, 50)), 30),
    ("pure", "outputs/ubfc_pure/remotessm_ubfc_pure_best.pth", 32, "cardio_v5_bvphr",
     list(range(109, 111)), 30),
    ("mmpd", "outputs/mmpd_dual_pt_v30/remotessm_mmpd_dual_pt_v30_best.pth", 64, "cardio_v6_roi64",
     list(range(78, 84)), 30),
]


def run_job(name, model_path, roi, cache_v, subjects, step):
    config.TRAIN_CONFIG["FACE_ROI_SIZE"] = roi
    config.TRAIN_CONFIG["CACHE_VERSION"] = cache_v
    config.TRAIN_CONFIG["VAL_STEP"] = step

    model = RemoteSSM(d_model=256, seq_len=300, n_ssm_blocks=2,
                      roi_size=roi, dropout=0.4, fs=30.0, use_fdf=True).to(device)
    ckpt = safe_torch_load(model_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    ext = DifferentiablePhysioExtractor(fs=30.0).to(device)

    ds = TitanrPPGDataset(root_dir=config.TRAIN_CONFIG["DATASET_ROOT"], seq_len=300,
                          step=step, subject_ids=subjects, split_name="test")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    preds, refs, tstamps, subjs = [], [], [], []
    with torch.no_grad():
        for b in loader:
            f = b["face_roi"].to(device)
            xr = b["input_real"].to(device)
            xi = b["input_imag"].to(device)
            thr = b["target_hr"].cpu().numpy()
            t0 = b.get("meta_start_idx", torch.zeros(len(thr))).numpy() / 30.0
            sb = b.get("subject_id", torch.zeros(len(thr))).numpy()
            bp, _ = model(face_roi=f, x_real=xr, x_imag=xi, mode="finetune")
            hp = ext(torch.clamp(bp, -3.0, 3.0), freq_range=(0.7, 3.0), temperature=0.05).cpu().numpy()
            preds.extend(hp.flatten().tolist())
            refs.extend(thr.flatten().tolist())
            tstamps.extend(t0.flatten().tolist())
            subjs.extend(sb.flatten().tolist())
    preds, refs = np.array(preds, np.float32), np.array(refs, np.float32)
    tstamps, subjs = np.array(tstamps, np.float32), np.array(subjs)
    m = np.isfinite(preds) & np.isfinite(refs) & (preds > 30) & (preds < 200) & (refs > 30) & (refs < 200)
    np.savez(os.path.join(OUT, f"{name}.npz"),
             pred=preds[m], ref=refs[m], t=tstamps[m], subj=subjs[m])
    print(f"[{name}] saved {m.sum()} pts (from {len(preds)}) | "
          f"MAE={np.mean(np.abs(preds[m] - refs[m])):.2f}")


if __name__ == "__main__":
    for job in JOBS:
        run_job(*job)
