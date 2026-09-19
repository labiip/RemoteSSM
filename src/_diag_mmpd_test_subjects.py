"""MMPD test 逐受试者诊断: 每个 test subject 单独评估. 用法: python3 _diag_mmpd_test_subjects.py"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
config.TRAIN_CONFIG["VAL_STEP"] = 300

from model import RemoteSSM
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor, safe_torch_load

device = torch.device("cuda")
seq_len = 300
roi_size = 64
d_model = 256
model = RemoteSSM(d_model=d_model, seq_len=seq_len, n_ssm_blocks=2,
                  roi_size=roi_size, dropout=0.4, fs=30.0, use_fdf=True).to(device)
ckpt = safe_torch_load(os.path.join(PROJ, "outputs", "mmpd_facemain", "remotessm_mmpd_facemain_best.pth"), map_location=device)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()
ext = DifferentiablePhysioExtractor(fs=30.0).to(device)


def fft_hr(bvp):
    with torch.no_grad():
        h = ext(torch.clamp(bvp, -3.0, 3.0), freq_range=(0.7, 3.0), temperature=0.05)
    return h


for subj in range(78, 84):
    ds = TitanrPPGDataset(root_dir=config.TRAIN_CONFIG["DATASET_ROOT"], seq_len=seq_len,
                          step=300, subject_ids=[subj], split_name="test")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            f = b["face_roi"].to(device)
            xr = torch.zeros_like(b["input_real"]).to(device)
            xi = torch.zeros_like(b["input_imag"]).to(device)
            thr = b["target_hr"]
            bp, _ = model(face_roi=f, x_real=xr, x_imag=xi, mode="finetune")
            ps.extend(fft_hr(bp).cpu().numpy().flatten().tolist())
            ts.extend(thr.numpy().flatten().tolist())
    ps, ts = np.array(ps, np.float32), np.array(ts, np.float32)
    m = np.isfinite(ps) & np.isfinite(ts) & (ps > 30) & (ps < 200) & (ts > 30) & (ts < 200)
    pp, tt = ps[m], ts[m]
    if len(pp) < 2:
        print(f"subject{subj}: n={len(pp)} 无效")
        continue
    r = np.corrcoef(pp, tt)[0, 1]
    mae = np.mean(np.abs(pp - tt))
    print(f"subject{subj}: n={len(pp)} MAE={mae:.2f} r={r:+.3f} pred_mean={pp.mean():.1f} target_mean={tt.mean():.1f}")
