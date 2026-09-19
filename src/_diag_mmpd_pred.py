"""MMPD face_roi-only 模型预测分布诊断.

验证是否"均值预测"假象: pred 方差≈0 且集中在均值附近 → MAE 虚低, r 低.
用法: python3 _diag_mmpd_pred.py
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from model import RemoteSSM
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor, safe_torch_load


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(PROJ, "outputs", "mmpd_facemain", "remotessm_mmpd_facemain_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJ, "outputs", "mmpd_facemain", "remotessm_mmpd_facemain_final.pth")
    print("ckpt:", ckpt_path)

    model = RemoteSSM(d_model=256, seq_len=300, n_ssm_blocks=2,
                      roi_size=32, dropout=0.4, fs=30.0, use_fdf=True).to(device)
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = TitanrPPGDataset(root_dir=TRAIN_CONFIG["DATASET_ROOT"], seq_len=300, step=300,
                          subject_ids=list(range(78, 84)), split_name="test")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    extractor = DifferentiablePhysioExtractor(fs=30.0).to(device)

    preds, gts = [], []
    preds05, gts2 = [], []
    with torch.no_grad():
        for b in loader:
            face = b["face_roi"].to(device)
            xr = b["input_real"].to(device)
            xi = b["input_imag"].to(device)
            thr = b["target_hr"].cpu().numpy()
            bvp, _ = model(face_roi=face, x_real=xr, x_imag=xi, mode="finetune")
            hp05 = extractor(torch.clamp(bvp, -3.0, 3.0),
                             freq_range=PHYSIO_CONFIG["HR_BAND"], temperature=0.05).cpu().numpy()
            hp5 = extractor(torch.clamp(bvp, -3.0, 3.0),
                            freq_range=PHYSIO_CONFIG["HR_BAND"], temperature=0.5).cpu().numpy()
            preds.extend(hp05.flatten().tolist())
            preds05.extend(hp5.flatten().tolist())
            gts.extend(thr.flatten().tolist())

    preds = np.array(preds, np.float32)
    preds05 = np.array(preds05, np.float32)
    gts = np.array(gts, np.float32)
    print(f"n={len(preds)}")
    print(f"GT:        mean={gts.mean():.2f} std={gts.std():.2f}")
    print(f"Pred(t=0.05): mean={preds.mean():.2f} std={preds.std():.2f} MAE={np.mean(np.abs(preds-gts)):.2f} r={np.corrcoef(preds,gts)[0,1]:.3f}")
    print(f"Pred(t=0.5):  mean={preds05.mean():.2f} std={preds05.std():.2f} MAE={np.mean(np.abs(preds05-gts)):.2f} r={np.corrcoef(preds05,gts)[0,1]:.3f}")


if __name__ == "__main__":
    main()
