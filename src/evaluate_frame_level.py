"""
RemoteSSM 逐帧 HR 轨迹评估脚本 (文献标准口径)

与文献一致的 Pearson 口径: 逐帧/逐窗口连续 HR 轨迹 vs GT 逐帧 HR 序列。
UBFC 的 ground_truth.txt 提供逐帧瞬时心率 (bvp, hr, time)，本脚本:
  1. 对每个测试受试者整段视频做滑动窗口推理 (step 秒级)
  2. 每个窗口由 BVP 波形 FFT 提取一个 HR 值
  3. 与窗口中心的 GT 逐帧 hr 对齐, 形成 HR 时间轨迹
  4. 计算 MAE / RMSE / Pearson / Acc<3/5/8

用法:
  python evaluate_frame_level.py                    # 默认 best checkpoint
  python evaluate_frame_level.py xxx.pth 30         # 指定权重 + step
"""

import os
import re
import sys
import numpy as np
import torch

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from model import RemoteSSM
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor, safe_torch_load


def _valid_mask(gt, pred, hr_min=40.0, hr_max=180.0):
    gt, pred = np.asarray(gt, np.float32), np.asarray(pred, np.float32)
    return (np.isfinite(gt) & np.isfinite(pred)
            & (gt >= hr_min) & (gt <= hr_max)
            & (pred >= hr_min) & (pred <= hr_max))


def evaluate_frame_level(model_path=None, step=30, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL-FRAME] RemoteSSM | Device: {device} | step={step}")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(project_dir, "remotessm_best.pth")
    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found: {model_path}")
        return

    seq_len = COMMON_CONFIG.get("SEQ_LEN", 300)
    d_model = COMMON_CONFIG.get("D_MODEL", 256)
    fs = COMMON_CONFIG.get("FS", 30.0)
    n_ssm_blocks = COMMON_CONFIG.get("N_SSM_BLOCKS", 2)
    roi_size = TRAIN_CONFIG.get("FACE_ROI_SIZE", 32)
    dropout = TRAIN_CONFIG.get("DROPOUT", 0.15)
    test_subjects = set(TRAIN_CONFIG["TEST_SUBJECTS"])
    root_dir = TRAIN_CONFIG["DATASET_ROOT"]

    model = RemoteSSM(d_model=d_model, seq_len=seq_len, n_ssm_blocks=n_ssm_blocks,
                      roi_size=roi_size, dropout=dropout, fs=fs, use_fdf=True).to(device)
    ckpt = safe_torch_load(model_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)

    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Loaded: {os.path.basename(model_path)} | Params: {n_p:.2f}M")

    # 实例化 dataset 以复用 face_roi_dict / bgr_dict / 时间对齐逻辑
    ds = TitanrPPGDataset(root_dir=root_dir, seq_len=seq_len, step=step,
                          subject_ids=test_subjects, split_name="test_frame")

    folders = [f for f in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, f))
               and not f.startswith("offline_cache_dir")]

    all_pred, all_gt = [], []
    all_head_pred = []
    n_subj = 0

    for folder in sorted(folders):
        sid = ds._extract_subject_id(folder)
        if sid is None or sid not in test_subjects:
            continue
        folder_path = os.path.join(root_dir, folder)

        gt_path = os.path.join(folder_path, "ground_truth.txt")
        if not os.path.exists(gt_path):
            print(f"[WARN] 无 GT: {folder}")
            continue

        bgr = ds.bgr_dict.get(folder)
        roi = ds.face_roi_dict.get(folder)
        if bgr is None or roi is None:
            print(f"[WARN] 无特征: {folder}")
            continue

        try:
            gt_data = np.loadtxt(gt_path)
        except Exception as e:
            print(f"[WARN] GT 解析失败 {folder}: {e}")
            continue

        if gt_data.ndim != 2:
            continue
        if gt_data.shape[0] == 3:
            gt_bvp, gt_hr, gt_time = gt_data[0], gt_data[1], gt_data[2]
        elif gt_data.shape[1] == 3:
            gt_bvp, gt_hr, gt_time = gt_data[:, 0], gt_data[:, 1], gt_data[:, 2]
        else:
            continue

        N = min(len(bgr), len(roi))
        bgr = bgr[:N].astype(np.float32)
        roi = roi[:N]

        # GT 逐帧 hr 对齐到视频时间轴 (长度 = N)
        aligned_bvp, aligned_hr, aligned_time = ds._align_gt_to_video_timeline(
            gt_bvp, gt_hr, gt_time, N)

        starts = list(range(0, N - seq_len + 1, step))
        centers = [s + seq_len // 2 for s in starts]
        n_win = len(starts)
        if n_win == 0:
            continue

        n_subj += 1
        for b in range(0, n_win, batch_size):
            b_starts = starts[b:b + batch_size]

            faces, xrs, xis, gt_bvps = [], [], [], []
            for s in b_starts:
                e = s + seq_len
                faces.append(roi[s:e].astype(np.float32) / 255.0)
                bgr_win = bgr[s:e] / 128.0 - 1.0
                xrs.append(bgr_win[:, 1:2].astype(np.float32))            # G
                xis.append((bgr_win[:, 0:1] - bgr_win[:, 2:3]).astype(np.float32))  # B-R
                gt_bvps.append(aligned_bvp[s:e].astype(np.float32))       # GT BVP 波形

            face_t = torch.from_numpy(np.stack(faces)).to(device)
            xr_t = torch.from_numpy(np.stack(xrs)).to(device)
            xi_t = torch.from_numpy(np.stack(xis)).to(device)
            gt_bvp_t = torch.from_numpy(np.stack(gt_bvps)).to(device)

            with torch.no_grad():
                bvp_p, hr_head_p = model(face_roi=face_t, x_real=xr_t, x_imag=xi_t, mode='finetune')
                # GT 与预测都用同一 FFT 提取器, 保证 HR 定义一致 (文献口径)
                hr_pred = extractor(torch.clamp(bvp_p, -3.0, 3.0),
                                    freq_range=PHYSIO_CONFIG["HR_BAND"], temperature=0.05)
                hr_gt = extractor(torch.clamp(gt_bvp_t, -3.0, 3.0),
                                  freq_range=PHYSIO_CONFIG["HR_BAND"], temperature=0.05)

            hr_pred = hr_pred.cpu().numpy().reshape(-1)
            hr_gt = hr_gt.cpu().numpy().reshape(-1)
            hr_head_pred = hr_head_p.cpu().numpy().reshape(-1)

            all_pred.extend(hr_pred.tolist())
            all_gt.extend(hr_gt.tolist())
            all_head_pred.extend(hr_head_pred.tolist())

    all_pred = np.asarray(all_pred, np.float32)
    all_gt = np.asarray(all_gt, np.float32)
    all_head_pred = np.asarray(all_head_pred, np.float32)

    print("\n" + "=" * 60)
    print("  RemoteSSM 逐帧 HR 轨迹评估 (文献口径)")
    print("=" * 60)
    print(f"  受试者数: {n_subj} | 窗口数: {len(all_gt)} (step={step}帧 ≈ {step/fs:.1f}s)")
    print(f"  GT  HR: mean={all_gt.mean():.2f} std={all_gt.std():.2f} min={all_gt.min():.2f} max={all_gt.max():.2f}")
    print("-" * 60)

    def _report(name, p):
        m = _valid_mask(all_gt, p)
        pp, tt = p[m], all_gt[m]
        if len(pp) < 2:
            print(f"  [{name}] 有效样本不足")
            return
        mae = np.mean(np.abs(pp - tt))
        rmse = np.sqrt(np.mean((pp - tt) ** 2))
        pear = float(np.corrcoef(pp, tt)[0, 1])
        a3 = np.mean(np.abs(pp - tt) <= 3) * 100
        a5 = np.mean(np.abs(pp - tt) <= 5) * 100
        a8 = np.mean(np.abs(pp - tt) <= 8) * 100
        print(f"\n  [{name}]")
        print(f"  mean={pp.mean():.2f} std={pp.std():.2f} min={pp.min():.2f} max={pp.max():.2f}")
        print(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  Pearson: {pear:.4f}")
        print(f"  Acc<3: {a3:.1f}%  Acc<5: {a5:.1f}%  Acc<8: {a8:.1f}%")

    _report("BVP FFT (文献标准)", all_pred)
    _report("HR Head 回归", all_head_pred)
    print("=" * 60)


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else None
    st = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    evaluate_frame_level(mp, st)
