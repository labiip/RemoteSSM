"""补两个论文增强评估:
1. UBFC-only 模型 → MMPD 全部 33 人 (跨域: 联合训练迁移收益对比)
2. MMPD 双路模型逐受试者分析 (失败模式)
用法: CUDA_VISIBLE_DEVICES=1 python3 _eval_supp.py
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
from evaluate import evaluate
from model import RemoteSSM
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor, safe_torch_load

device = torch.device("cuda")
MMPD_ALL = list(range(51, 84))
MMPD_TEST = list(range(78, 84))

# 1) UBFC-only 跨域
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 32
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v5_bvphr"
config.TRAIN_CONFIG["VAL_STEP"] = 30
print("\n" + "=" * 56 + "\n[UBFC-only model → MMPD ALL] (leave_mmpd_single)\n" + "=" * 56)
evaluate(model_path=os.path.join(PROJ, "outputs", "ubfc_only", "remotessm_ubfc_only_best.pth"),
         test_subjects=MMPD_ALL)

# 2) MMPD 双路逐受试者
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
model = RemoteSSM(d_model=256, seq_len=300, n_ssm_blocks=2,
                  roi_size=64, dropout=0.4, fs=30.0, use_fdf=True).to(device)
ckpt = safe_torch_load(os.path.join(PROJ, "outputs", "mmpd_dual_pt", "remotessm_mmpd_dual_pt_best.pth"),
                       map_location=device)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()
ext = DifferentiablePhysioExtractor(fs=30.0).to(device)

print("\n" + "=" * 56 + "\n[MMPD dual 逐受试者]\n" + "=" * 56)
for subj in MMPD_TEST:
    ds = TitanrPPGDataset(root_dir=config.TRAIN_CONFIG["DATASET_ROOT"], seq_len=300,
                          step=30, subject_ids=[subj], split_name="test")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            f = b["face_roi"].to(device)
            xr = b["input_real"].to(device)
            xi = b["input_imag"].to(device)
            bp, _ = model(face_roi=f, x_real=xr, x_imag=xi, mode="finetune")
            hp = ext(torch.clamp(bp, -3.0, 3.0), freq_range=(0.7, 3.0), temperature=0.05).cpu().numpy()
            ps.extend(hp.flatten().tolist())
            ts.extend(b["target_hr"].numpy().flatten().tolist())
    ps, ts = np.array(ps, np.float32), np.array(ts, np.float32)
    m = np.isfinite(ps) & np.isfinite(ts) & (ps > 30) & (ps < 200) & (ts > 30) & (ts < 200)
    pp, tt = ps[m], ts[m]
    r = np.corrcoef(pp, tt)[0, 1] if len(pp) > 2 else float("nan")
    mae = np.mean(np.abs(pp - tt))
    print(f"subject{subj}: n={len(pp)} MAE={mae:.2f} r={r:+.3f} pred_mean={pp.mean():.1f} ref_mean={tt.mean():.1f}")
