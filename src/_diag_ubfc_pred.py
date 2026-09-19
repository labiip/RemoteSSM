"""UBFC+PURE 模型在 UBFC test 的复核: temperature=0.05 vs 0.5.

用法: python3 _diag_ubfc_pred.py
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
    ckpt_path = os.path.join(PROJ, "outputs", "ubfc_pure_quick", "remotessm_ubfc_pure_quick_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_final.pth")
    print("ckpt:", ckpt_path)

    model = RemoteSSM(d_model=256, seq_len=300, n_ssm_blocks=2,
                      roi_size=32, dropout=0.4, fs=30.0, use_fdf=True).to(device)
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    for name, subj in [("UBFC test 40-49", list(range(40, 50))),
                       ("PURE test 109-110", list(range(109, 111)))]:
        ds = TitanrPPGDataset(root_dir=TRAIN_CONFIG["DATASET_ROOT"], seq_len=300, step=300,
                              subject_ids=subj, split_name="test")
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        extractor = DifferentiablePhysioExtractor(fs=30.0).to(device)
        p05, p5, gt = [], [], []
        with torch.no_grad():
            for b in loader:
                face = b["face_roi"].to(device)
                xr = b["input_real"].to(device)
                xi = b["input_imag"].to(device)
                thr = b["target_hr"].cpu().numpy()
                bvp, _ = model(face_roi=face, x_real=xr, x_imag=xi, mode="finetune")
                p05.extend(extractor(torch.clamp(bvp, -3.0, 3.0), freq_range=PHYSIO_CONFIG["HR_BAND"], temperature=0.05).cpu().numpy().flatten().tolist())
                p5.extend(extractor(torch.clamp(bvp, -3.0, 3.0), freq_range=PHYSIO_CONFIG["HR_BAND"], temperature=0.5).cpu().numpy().flatten().tolist())
                gt.extend(thr.flatten().tolist())
        p05 = np.array(p05, np.float32); p5 = np.array(p5, np.float32); gt = np.array(gt, np.float32)
        print(f"[{name}] n={len(gt)}")
        print(f"  GT: mean={gt.mean():.2f} std={gt.std():.2f}")
        print(f"  t=0.05: mean={p05.mean():.2f} std={p05.std():.2f} MAE={np.mean(np.abs(p05-gt)):.2f} r={np.corrcoef(p05,gt)[0,1]:.3f}")
        print(f"  t=0.5:  mean={p5.mean():.2f} std={p5.std():.2f} MAE={np.mean(np.abs(p5-gt)):.2f} r={np.corrcoef(p5,gt)[0,1]:.3f}")


if __name__ == "__main__":
    main()
