"""
RemoteSSM 评估脚本

在测试集上评估: MAE, RMSE, Pearson, Acc<3/5/8BPM
使用 BVP 衍生 HR 标签 (文献标准方式)

用法:
  python evaluate.py                          # 默认 best checkpoint
  python evaluate.py remotessm_best.pth     # 指定权重
"""

import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from model import RemoteSSM
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor, safe_torch_load


def _valid_mask(gt, pred, hr_min=30.0, hr_max=200.0):
    gt, pred = np.asarray(gt, np.float32), np.asarray(pred, np.float32)
    return (np.isfinite(gt) & np.isfinite(pred)
            & (gt >= hr_min) & (gt <= hr_max)
            & (pred >= hr_min) & (pred <= hr_max))


def evaluate(model_path=None, test_subjects=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] RemoteSSM | Device: {device}")

    project_dir = os.path.dirname(os.path.abspath(__file__))

    if model_path is None:
        model_path = os.path.join(project_dir, "remotessm_best.pth")
    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found: {model_path}")
        return

    # ── Config ──
    seq_len = COMMON_CONFIG.get("SEQ_LEN", 300)
    d_model = COMMON_CONFIG.get("D_MODEL", 256)
    fs = COMMON_CONFIG.get("FS", 30.0)
    n_ssm_blocks = COMMON_CONFIG.get("N_SSM_BLOCKS", 2)
    roi_size = TRAIN_CONFIG.get("FACE_ROI_SIZE", 32)
    dropout = TRAIN_CONFIG.get("DROPOUT", 0.15)
    val_step = TRAIN_CONFIG.get("VAL_STEP", 300)
    batch_size = TRAIN_CONFIG.get("BATCH_SIZE", 8)

    if test_subjects is None:
        test_subjects = TRAIN_CONFIG["TEST_SUBJECTS"]

    # ── Model ──
    model = RemoteSSM(d_model=d_model, seq_len=seq_len, n_ssm_blocks=n_ssm_blocks,
                      roi_size=roi_size, dropout=dropout, fs=fs, use_fdf=True).to(device)
    ckpt = safe_torch_load(model_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    m, u = model.load_state_dict(sd, strict=False)
    if m: print(f"[WARN] Missing {len(m)} keys")
    model.eval()

    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Loaded: {os.path.basename(model_path)} | Params: {n_p:.2f}M")

    # ── Dataset ──
    ds = TitanrPPGDataset(root_dir=TRAIN_CONFIG["DATASET_ROOT"],
                          seq_len=seq_len, step=val_step,
                          subject_ids=test_subjects, split_name="test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f"[INFO] Test: {len(ds)} samples | subjects {test_subjects}")

    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)

    all_fft_p, all_fft_t = [], []
    all_head_p, all_head_t = [], []

    with torch.no_grad():
        for batch in loader:
            face = batch['face_roi'].to(device)
            xr = batch['input_real'].to(device)
            xi = batch['input_imag'].to(device)
            thr = batch['target_hr'].cpu().numpy()

            bvp_p, hr_p = model(face_roi=face, x_real=xr, x_imag=xi, mode='finetune')
            fft_p = extractor(torch.clamp(bvp_p, -3.0, 3.0),
                              freq_range=PHYSIO_CONFIG["HR_BAND"],
                              temperature=0.05).cpu().numpy()
            head_p = hr_p.cpu().numpy()

            all_fft_p.extend(fft_p.flatten().tolist())
            all_fft_t.extend(thr.flatten().tolist())
            all_head_p.extend(head_p.flatten().tolist())
            all_head_t.extend(thr.flatten().tolist())

    all_fft_p = np.array(all_fft_p, np.float32)
    all_fft_t = np.array(all_fft_t, np.float32)
    all_head_p = np.array(all_head_p, np.float32)
    all_head_t = np.array(all_head_t, np.float32)

    print("\n" + "=" * 56)
    print("  RemoteSSM Test Report")
    print("=" * 56)

    for name, p, t in [("FFT(BVP) vs BVP-HR", all_fft_p, all_fft_t),
                        ("HR Head  vs BVP-HR", all_head_p, all_head_t)]:
        m = _valid_mask(t, p)
        pp, tt = p[m], t[m]
        if len(pp) < 2:
            print(f"\n  [{name}] Too few valid samples ({len(pp)})")
            continue

        mae = np.mean(np.abs(pp - tt))
        rmse = np.sqrt(np.mean((pp - tt) ** 2))
        std = np.std(np.abs(pp - tt))
        pear = np.corrcoef(pp, tt)[0, 1] if len(pp) > 1 else 0
        a3 = np.mean(np.abs(pp - tt) <= 3) * 100
        a5 = np.mean(np.abs(pp - tt) <= 5) * 100
        a8 = np.mean(np.abs(pp - tt) <= 8) * 100

        print(f"\n  [{name}]")
        print(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  StdErr: {std:.2f}  Pearson: {pear:.4f}")
        print(f"  Acc<3: {a3:.1f}%  Acc<5: {a5:.1f}%  Acc<8: {a8:.1f}%")
        print(f"  Valid: {len(pp)}/{len(p)}")

    print("\n" + "-" * 56)
    print(f"  Label: {'BVP-derived' if TRAIN_CONFIG.get('USE_BVP_DERIVED_HR_LABEL', True) else 'ECG gold'}")
    print("-" * 56)


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(mp)
