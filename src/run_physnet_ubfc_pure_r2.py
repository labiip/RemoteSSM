"""PhysNet 同协议复现, run 2 (r2: 弱增强 + 无 wd, 验证 r1 不稳定是否由超参导致).

与 run_physnet_ubfc_pure.py 完全同协议 (UBFC+PURE 联合划分 / step-30 评测 /
negPearson 波形监督 / Adam lr=1e-3 / warmup+cosine / AMP off)。
r2 相对 r1 的差异 (r1 PURE test 17.29 且 val 剧烈震荡):
  - face 增强 brightness/contrast 0.4 -> 0.15, 像素噪声 0.06 -> 0.02 (过强增强导致逐轮权重抖动)
  - weight decay 5e-3 -> 0.0 (0.77M 小模型 + 40 epoch 无需强正则)
输出独立目录 outputs/physnet_ubfc_pure_r2, 不覆盖 r1。
"""
import os
import sys
import math
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# ---- 协议参数 (与 run_ubfc_pure.py 主实验一致; 评测固定 step30) ----
config.TRAIN_CONFIG["EPOCHS"] = 40
config.TRAIN_CONFIG["TRAIN_STEP"] = 10          # 训练窗采样步长 (复用已有缓存)
config.TRAIN_CONFIG["VAL_STEP"] = 30            # 评测/验证: 10s 窗, step 30 帧
config.TRAIN_CONFIG["BATCH_SIZE"] = 16
config.TRAIN_CONFIG["LEARNING_RATE"] = 1e-3     # PhysNet 原论文 Adam lr
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [40]

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from dataset import TitanrPPGDataset
from physnet_model import PhysNet3DCNN
from utils import DifferentiablePhysioExtractor, safe_torch_load
from train import pearson_loss, face_roi_augmentation, set_seed, get_num_workers

TRAIN = list(range(1, 35)) + list(range(101, 107))   # UBFC 1-34 + PURE 101-106
VAL = list(range(35, 40)) + list(range(107, 109))    # UBFC 35-39 + PURE 107-108
TEST_UBFC = list(range(40, 50))
TEST_PURE = list(range(109, 111))

OUT_DIR = os.path.join(PROJ, "outputs", "physnet_ubfc_pure_r2")
TAG = "physnet_ubfc_pure_r2"
BEST_PATH = os.path.join(OUT_DIR, f"{TAG}_best.pth")
RESUME_PATH = os.path.join(OUT_DIR, f"{TAG}_resume.pth")
LOG_PATH = os.path.join(OUT_DIR, f"{TAG}_train.log")


# ═══════════════════ 训练 ═══════════════════

def train_model():
    set_seed(COMMON_CONFIG.get("SEED", 42))
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = int(COMMON_CONFIG["SEQ_LEN"])
    fs = float(COMMON_CONFIG["FS"])
    epochs = int(TRAIN_CONFIG["EPOCHS"])
    lr = float(TRAIN_CONFIG["LEARNING_RATE"])
    batch_size = int(TRAIN_CONFIG["BATCH_SIZE"])
    wd = 0.0   # r2: 去掉 weight decay (0.77M 小模型 40 epoch 无需强正则)
    grad_clip = float(TRAIN_CONFIG.get("GRAD_CLIP", 1.0))
    num_workers = get_num_workers(TRAIN_CONFIG.get("NUM_WORKERS", 4))
    lr_warmup = int(TRAIN_CONFIG.get("LR_WARMUP_EPOCHS", 5))
    dataset_root = TRAIN_CONFIG["DATASET_ROOT"]

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    # 数据 (与 RemoteSSM-v3 主实验同划分同缓存)
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

    model = PhysNet3DCNN(frames=seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"PhysNet-3DCNN (r2) | Params: {n_params:.2f}M | Device: {device}")
    logging.info(f"Train {len(train_ds)} (step10) | Val {len(val_ds)} (step30)")
    logging.info(f"Epochs {epochs} | Adam lr={lr} wd={wd} | warmup {lr_warmup}+cosine | AMP off")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def _lr_lambda(ep):
        if ep < lr_warmup:
            return (ep + 1) / max(1, lr_warmup)
        prog = (ep - lr_warmup) / max(1, epochs - lr_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)

    fa_b = 0.15   # r2: 弱化 face 增强 (0.4 过强导致 val 震荡)
    fa_c = 0.15
    fa_n = 0.02

    # Resume
    best_val_mae = float("inf")
    start_epoch = 1
    if os.path.exists(RESUME_PATH):
        ck = safe_torch_load(RESUME_PATH, map_location=device)
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
        df = db["face_roi"][:2].to(device)
        model.eval()
        dy = model(df)
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
            face = batch["face_roi"].to(device)
            t_bvp = batch["target_bvp"].to(device)
            face_a = face_roi_augmentation(face, fa_b, fa_c, fa_n)

            bvp_p = model(face_a)
            loss = pearson_loss(bvp_p, t_bvp)          # 原论文 trend-consistency (负 Pearson)

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

        # Validation (FFT extractor, step30) — 与主实验同口径
        model.eval()
        val_errs, val_hp, val_ht = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                f = batch["face_roi"].to(device)
                thr = batch["target_hr"].cpu().numpy()
                bp = model(f)
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
            torch.save({"model_state_dict": model.state_dict()}, BEST_PATH)
            logging.info(f"  ✓ Best: {val_mae:.2f} BPM")

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "best_val_mae": best_val_mae}, RESUME_PATH)

    logging.info(f"Done. Best Val FFT MAE: {best_val_mae:.2f} BPM")


# ═══════════════════ 评测 (同 evaluate 口径) ═══════════════════

def _valid_mask(gt, pred, hr_min=30.0, hr_max=200.0):
    gt, pred = np.asarray(gt, np.float32), np.asarray(pred, np.float32)
    return ((np.isfinite(gt) & np.isfinite(pred))
            & (gt >= hr_min) & (gt <= hr_max)
            & (pred >= hr_min) & (pred <= hr_max))


def evaluate(model_path, test_subjects, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found: {model_path}")
        return
    seq_len = int(COMMON_CONFIG["SEQ_LEN"])
    fs = float(COMMON_CONFIG["FS"])

    model = PhysNet3DCNN(frames=seq_len).to(device)
    ck = safe_torch_load(model_path, map_location=device)
    sd = ck.get("model_state_dict", ck)
    m, u = model.load_state_dict(sd, strict=False)
    if m:
        print(f"[WARN] Missing {len(m)} keys")
    model.eval()
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] {name} | loaded {os.path.basename(model_path)} | Params {n_p:.2f}M")

    ds = TitanrPPGDataset(root_dir=TRAIN_CONFIG["DATASET_ROOT"], seq_len=seq_len,
                          step=int(TRAIN_CONFIG["VAL_STEP"]),
                          subject_ids=test_subjects, split_name="test")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[INFO] Test {name}: {len(ds)} samples | subjects {test_subjects}")

    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            face = batch["face_roi"].to(device)
            thr = batch["target_hr"].cpu().numpy()
            bp = model(face)
            hp = extractor(torch.clamp(bp, -3.0, 3.0),
                           freq_range=PHYSIO_CONFIG["HR_BAND"],
                           temperature=0.05).cpu().numpy()
            all_p.extend(hp.flatten().tolist())
            all_t.extend(thr.flatten().tolist())

    p = np.array(all_p, np.float32)
    t = np.array(all_t, np.float32)
    msk = _valid_mask(t, p)
    pp, tt = p[msk], t[msk]
    mae = np.mean(np.abs(pp - tt))
    rmse = np.sqrt(np.mean((pp - tt) ** 2))
    pear = np.corrcoef(pp, tt)[0, 1] if len(pp) > 1 else 0.0
    print(f"\n[{name}] PhysNet (r2, same protocol, step-30)")
    print(f"  MAE: {mae:.2f}  RMSE: {rmse:.2f}  Pearson: {pear:.4f}")
    print(f"  Acc<3: {np.mean(np.abs(pp - tt) <= 3) * 100:.1f}%  "
          f"Acc<5: {np.mean(np.abs(pp - tt) <= 5) * 100:.1f}%  "
          f"Acc<8: {np.mean(np.abs(pp - tt) <= 8) * 100:.1f}%")
    print(f"  Valid: {len(pp)}/{len(p)}")
    return mae, rmse, pear


if __name__ == "__main__":
    train_model()
    print("\n" + "=" * 56)
    print("  PhysNet r2 test report (same protocol)")
    print("=" * 56)
    evaluate(BEST_PATH, TEST_UBFC, "UBFC test 40-49")
    evaluate(BEST_PATH, TEST_PURE, "PURE test 109-110")
