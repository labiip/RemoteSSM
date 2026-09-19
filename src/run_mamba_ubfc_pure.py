"""Mamba 系同协议复现 (UBFC+PURE 联合训练, subject-independent).

与 RemoteSSM 主实验 / PhysNet 复现完全相同的协议:
  - 同一数据管道: 32px face ROI 缓存 cardio_v5_bvphr, seq 300
  - 同一数据划分: train=UBFC 1-34 + PURE 101-106 (40 subj),
                  val  =UBFC 35-39 + PURE 107-108,
                  test =UBFC 40-49 / PURE 109-110
  - 同一评测口径: 10s 窗 step30 (VAL_STEP=30), 0.7-3.0 Hz,
                  BVP 衍生 HR 标签, soft-argmax FFT(BVP) 提取 HR
  - 训练目标: 负 Pearson (两方法原论文一致)
  - 输入: face_roi 原样接入; PhysMamba 采用官方 DiffNormalized 帧差分预处理,
          RhythmMamba 采用官方 Standardized(z-score) 预处理

模型忠实官方:
  - PhysMamba (Luo et al., CCBR 2024): SlowFast Bi-Mamba, 官方模型文件,
    Bi-Mamba 由 vendor_bimamba.MambaBi 复刻(内核即服务器自编译 mamba_ssm)
  - RhythmMamba (Zou et al., AAAI 2025): 官方 RhythmMamba.py (embed96 depth24)

用法 (服务器, venv):
  CUDA_VISIBLE_DEVICES=X nohup ~/mamba_env/bin/python -u \
      run_mamba_ubfc_pure.py --model physmamba > physmamba_ubfc_pure.out 2>&1 &
"""
import argparse
import os
import sys
import math
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

torch.backends.cudnn.enabled = False  # cuDNN9 无 Pascal(sm61) 3D-conv engine, 走原生实现

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

config.TRAIN_CONFIG["EPOCHS"] = 40
config.TRAIN_CONFIG["TRAIN_STEP"] = 10
config.TRAIN_CONFIG["VAL_STEP"] = 30
config.TRAIN_CONFIG["BATCH_SIZE"] = 8
config.TRAIN_CONFIG["LEARNING_RATE"] = 1e-3
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [40]

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor, safe_torch_load
from train import pearson_loss, face_roi_augmentation, set_seed, get_num_workers

TRAIN = list(range(1, 35)) + list(range(101, 107))
VAL = list(range(35, 40)) + list(range(107, 109))
TEST_UBFC = list(range(40, 50))
TEST_PURE = list(range(109, 111))

SEQ = int(COMMON_CONFIG["SEQ_LEN"])  # 300


# ═════════════════ 模型装配 ═════════════════

def build_model(name, seq_len):
    if name == "physmamba":
        # Bi-Mamba 需要先替换 mamba_ssm.Mamba 顶层导出
        import mamba_ssm
        from vendor_bimamba import MambaBi
        mamba_ssm.Mamba = MambaBi
        from physmamba_model import PhysMamba
        return PhysMamba(frames=seq_len)
    elif name == "rhythmmamba":
        from rhythmmamba_model import RhythmMamba
        return RhythmMamba(depth=24, embed_dim=96)
    raise ValueError(name)


# ═════════════════ 输入预处理 (与官方一致) ═════════════════

def physmamba_input(face, device):
    """face: (B,T,H,W,3) float. -> (B,3,T,H,W) DiffNormalized."""
    x = face.permute(0, 4, 1, 2, 3).contiguous()          # (B,3,T,H,W)
    xf = x.float()
    x_next = torch.cat([xf[:, :, 1:], xf[:, :, -1:]], dim=2)
    diff = (x_next - xf) / (x_next + xf + 1e-7)            # 末帧 padding
    diff = diff / (diff.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6)
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    return diff.to(device)


def rhythm_input(face, device):
    """face: (B,T,H,W,3) float. -> (B,T,3,H,W) Standardized."""
    x = face.permute(0, 1, 4, 2, 3).contiguous()          # (B,T,3,H,W)
    mu = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    sd = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6
    return ((x - mu) / sd).to(device)


# ═════════════════ 训练 ═════════════════

def train_model(name):
    tag = f"{name}_ubfc_pure"
    out_dir = os.path.join(PROJ, "outputs", tag)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{tag}_best.pth")
    resume_path = os.path.join(out_dir, f"{tag}_resume.pth")
    log_path = os.path.join(out_dir, f"{tag}_train.log")

    set_seed(COMMON_CONFIG.get("SEED", 42))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = float(COMMON_CONFIG["FS"])
    seq_len = int(COMMON_CONFIG["SEQ_LEN"])
    epochs = int(TRAIN_CONFIG["EPOCHS"])
    lr = float(TRAIN_CONFIG["LEARNING_RATE"])
    batch_size = int(TRAIN_CONFIG["BATCH_SIZE"])
    wd = float(TRAIN_CONFIG.get("WEIGHT_DECAY", 0.0))
    grad_clip = float(TRAIN_CONFIG.get("GRAD_CLIP", 1.0))
    num_workers = get_num_workers(TRAIN_CONFIG.get("NUM_WORKERS", 4))
    lr_warmup = int(TRAIN_CONFIG.get("LR_WARMUP_EPOCHS", 5))
    dataset_root = TRAIN_CONFIG["DATASET_ROOT"]

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    train_ds = TitanrPPGDataset(root_dir=dataset_root, seq_len=seq_len,
                                step=int(TRAIN_CONFIG["TRAIN_STEP"]),
                                subject_ids=TRAIN, split_name="train")
    val_ds = TitanrPPGDataset(root_dir=dataset_root, seq_len=seq_len,
                              step=int(TRAIN_CONFIG["VAL_STEP"]),
                              subject_ids=VAL, split_name="val")
    assert len(train_ds) > 0 and len(val_ds) > 0, "Empty dataset!"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = build_model(name, seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"{name} | Params: {n_params:.2f}M | Device: {device}")
    logging.info(f"Train {len(train_ds)} (step10) | Val {len(val_ds)} (step30)")
    logging.info(f"Epochs {epochs} | Adam lr={lr} wd={wd} | warmup {lr_warmup}+cosine | AMP off | cudnn off")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def _lr_lambda(ep):
        if ep < lr_warmup:
            return (ep + 1) / max(1, lr_warmup)
        prog = (ep - lr_warmup) / max(1, epochs - lr_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)

    fa_b = float(TRAIN_CONFIG.get("FACE_AUG_BRIGHTNESS", 0.4))
    fa_c = float(TRAIN_CONFIG.get("FACE_AUG_CONTRAST", 0.4))

    to_in = physmamba_input if name == "physmamba" else rhythm_input

    # Resume
    best_val_mae = float("inf")
    start_epoch = 1
    if os.path.exists(resume_path):
        ck = safe_torch_load(resume_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        except Exception:
            logging.warning("scheduler state incompatible, restart LR schedule")
        start_epoch = int(ck["epoch"]) + 1
        best_val_mae = float(ck.get("best_val_mae", float("inf")))
        logging.info(f"Resumed at epoch {start_epoch} (best={best_val_mae:.2f})")

    # Pre-flight
    with torch.no_grad():
        db = next(iter(train_loader))
        xi = to_in(db["face_roi"][:2], device)
        model.eval()
        dy = model(xi)
        model.train()
        nan0 = bool(torch.isnan(dy).any())
        logging.info(f"Pre-flight BVP range [{dy.min().item():.3f}, {dy.max().item():.3f}] NaN={nan0}")
        if nan0:
            logging.error("NaN on first forward — aborting.")
            return

    nan_streak = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            face = batch["face_roi"]
            t_bvp = batch["target_bvp"].to(device)
            face_a = face_roi_augmentation(face, fa_b, fa_c, 0.06)
            xi = to_in(face_a, device)
            bvp_p = model(xi)
            loss = pearson_loss(bvp_p, t_bvp)

            if not torch.isfinite(loss):
                nan_streak += 1
                logging.warning(f"E{epoch} B{batch_idx}: NaN loss (streak={nan_streak})")
                if nan_streak > 50:
                    logging.error("Persistent NaN, stopping.")
                    return
                optimizer.zero_grad()
                continue
            nan_streak = 0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()

        # Validation (FFT extractor, step30)
        model.eval()
        val_errs, val_hp, val_ht = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                xi = to_in(batch["face_roi"], device)
                thr = batch["target_hr"].cpu().numpy()
                bp = model(xi)
                hp = extractor(torch.clamp(bp, -3.0, 3.0),
                               freq_range=PHYSIO_CONFIG["HR_BAND"],
                               temperature=0.05).cpu().numpy()
                val_errs.extend(np.abs(hp - thr).flatten().tolist())
                val_hp.extend(hp.flatten().tolist())
                val_ht.extend(thr.flatten().tolist())

        def _pearson(a, b):
            a = np.asarray(a, np.float32).flatten()
            b = np.asarray(b, np.float32).flatten()
            if len(a) < 2 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        val_mae = float(np.mean(val_errs)) if val_errs else 999.0
        val_r = _pearson(val_hp, val_ht)
        logging.info(f"Epoch {epoch:3d}/{epochs} | Loss {ep_loss / max(nb, 1):.4f} "
                     f"| Val FFT MAE {val_mae:.2f} | Val r(HR) {val_r:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            logging.info(f"  ✓ Best: {val_mae:.2f} BPM")

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "best_val_mae": best_val_mae}, resume_path)

    logging.info(f"Done. Best Val FFT MAE: {best_val_mae:.2f} BPM")


# ═════════════════ 评测 ═════════════════

def _valid_mask(gt, pred, hr_min=30.0, hr_max=200.0):
    gt, pred = np.asarray(gt, np.float32), np.asarray(pred, np.float32)
    return ((np.isfinite(gt) & np.isfinite(pred))
            & (gt >= hr_min) & (gt <= hr_max)
            & (pred >= hr_min) & (pred <= hr_max))


def evaluate(name, model_path, test_subjects, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found: {model_path}")
        return
    fs = float(COMMON_CONFIG["FS"])
    seq_len = int(COMMON_CONFIG["SEQ_LEN"])

    model = build_model(name, seq_len).to(device)
    ck = safe_torch_load(model_path, map_location=device)
    sd = ck.get("model_state_dict", ck)
    m, u = model.load_state_dict(sd, strict=False)
    if m:
        print(f"[WARN] Missing {len(m)} keys")
    model.eval()
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] {name} {label} | loaded {os.path.basename(model_path)} | Params {n_p:.2f}M")

    ds = TitanrPPGDataset(root_dir=TRAIN_CONFIG["DATASET_ROOT"], seq_len=seq_len,
                          step=int(TRAIN_CONFIG["VAL_STEP"]),
                          subject_ids=test_subjects, split_name="test")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[INFO] Test {label}: {len(ds)} samples")

    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)
    to_in = physmamba_input if name == "physmamba" else rhythm_input
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            xi = to_in(batch["face_roi"], device)
            thr = batch["target_hr"].cpu().numpy()
            bp = model(xi)
            hp = extractor(torch.clamp(bp, -3.0, 3.0),
                           freq_range=PHYSIO_CONFIG["HR_BAND"],
                           temperature=0.05).cpu().numpy()
            all_p.extend(hp.flatten().tolist())
            all_t.extend(thr.flatten().tolist())

    p = np.array(all_p, np.float32)
    t = np.array(all_t, np.float32)
    msk = _valid_mask(t, p)
    pp, tt = p[msk], t[msk]
    if len(pp) < 2:
        print(f"[{label}] insufficient valid samples: {len(pp)}/{len(p)}")
        return
    mae = np.mean(np.abs(pp - tt))
    rmse = np.sqrt(np.mean((pp - tt) ** 2))
    pear = np.corrcoef(pp, tt)[0, 1]
    print(f"\n[{label}] {name} (same protocol, step-30)")
    print(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  Pearson: {pear:.4f}")
    print(f"  Acc<3: {np.mean(np.abs(pp - tt) <= 3) * 100:.1f}%  "
          f"Acc<5: {np.mean(np.abs(pp - tt) <= 5) * 100:.1f}%  "
          f"Acc<8: {np.mean(np.abs(pp - tt) <= 8) * 100:.1f}%")
    print(f"  Valid: {len(pp)}/{len(p)}")
    return mae, rmse, pear


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["physmamba", "rhythmmamba"])
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数(用于快速试训)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.epochs:
        config.TRAIN_CONFIG["EPOCHS"] = args.epochs

    if args.eval_only:
        tag = f"{args.model}_ubfc_pure"
        evaluate(args.model, os.path.join(PROJ, "outputs", tag, f"{tag}_best.pth"),
                 TEST_UBFC, "UBFC test 40-49")
        evaluate(args.model, os.path.join(PROJ, "outputs", tag, f"{tag}_best.pth"),
                 TEST_PURE, "PURE test 109-110")
    else:
        train_model(args.model)
        print("\n" + "=" * 56)
        print(f"  {args.model} test report (same protocol)")
        print("=" * 56)
        tag = f"{args.model}_ubfc_pure"
        evaluate(args.model, os.path.join(PROJ, "outputs", tag, f"{tag}_best.pth"),
                 TEST_UBFC, "UBFC test 40-49")
        evaluate(args.model, os.path.join(PROJ, "outputs", tag, f"{tag}_best.pth"),
                 TEST_PURE, "PURE test 109-110")
