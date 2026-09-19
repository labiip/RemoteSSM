"""
Stage 1: 仿真去噪预训练 (RemoteSSM 专用)

用UBFC的GT BVP波形 + PhysioNoiseSimulator 生成72K训练对,
训练 RemoteSSM 在 denoise 模式下: noisy_bvp → clean_bvp.

COB 心脏振荡器在预训练阶段即锁定 0.7-3.0 Hz 心率频段,
使 FDF 权重也学到生理相关的频率增强模式。

数据: 49人 BVP 波形 × 20 噪声变体 ≈ 72K 训练对
时间: ~3 小时 (100 epochs, ~110 秒/epoch)
"""

import os
import random
import logging
import platform
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from contextlib import nullcontext

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import COMMON_CONFIG, TRAIN_CONFIG
from dataset import TitanrPPGDataset
from model import RemoteSSM
from noise_simulator import PhysioNoiseSimulator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_workers(requested):
    if requested is None or requested <= 0:
        return 0
    safe = min(int(requested), os.cpu_count() or 1)
    return min(safe, 4) if platform.system().lower() == "windows" else safe


def get_autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type if hasattr(device, 'type') else str(device), enabled=True)
    except Exception:
        if hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def create_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        class Dummy:
            def scale(self, loss): return loss
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
        return Dummy()


def pearson_loss(pred, target):
    pred, target = pred.view(pred.shape[0], -1), target.view(target.shape[0], -1)
    pm, tm = pred.mean(1, keepdim=True), target.mean(1, keepdim=True)
    pd, td = pred - pm, target - tm
    num = (pd * td).sum(1)
    den = (pd ** 2).sum(1).sqrt() * (td ** 2).sum(1).sqrt() + 1e-8
    return 1.0 - (num / den).mean()


def cardiac_freq_loss(pred, target, fs=30.0, fmin=0.7, fmax=3.0):
    p_f, t_f = pred.float(), target.float()
    pf = torch.abs(torch.fft.rfft(p_f, dim=1))
    tf = torch.abs(torch.fft.rfft(t_f, dim=1))
    n = pred.shape[1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs).to(pred.device)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return F.l1_loss(pf[:, mask], tf[:, mask]) if mask.any() else torch.tensor(0.0, device=pred.device)


class DenoisingCollator:
    def __init__(self, sim):
        self.sim = sim

    def __call__(self, batch):
        clean = torch.stack([item['target_bvp'] for item in batch])
        if clean.dim() == 3:
            clean = clean.squeeze(-1)
        noisy = self.sim(clean)
        clean = clean / (clean.std(dim=1, keepdim=True) + 1e-8)
        noisy = noisy / (noisy.std(dim=1, keepdim=True) + 1e-8)
        return {'noisy': noisy, 'clean': clean}


def main():
    set_seed(COMMON_CONFIG.get("SEED", 42))
    project_dir = os.path.dirname(os.path.abspath(__file__))

    log_path = os.path.join(project_dir, "pretrain_stage1_v3.log")
    ckpt_path = os.path.join(project_dir, "pretrain_stage1_v3.pth")
    resume_path = os.path.join(project_dir, "pretrain_stage1_v3_resume.pth")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"RemoteSSM Stage 1 | Device: {device}")

    seq_len = COMMON_CONFIG["SEQ_LEN"]
    d_model = COMMON_CONFIG["D_MODEL"]
    n_ssm_blocks = COMMON_CONFIG.get("N_SSM_BLOCKS", 2)
    fs = COMMON_CONFIG["FS"]
    n_epochs = 100
    batch_size = TRAIN_CONFIG["BATCH_SIZE"]

    dataset_root = TRAIN_CONFIG["DATASET_ROOT"]
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    # All 49 subjects for pretraining
    ds = TitanrPPGDataset(root_dir=dataset_root, seq_len=seq_len,
                          step=TRAIN_CONFIG["TRAIN_STEP"],
                          subject_ids=list(range(1, 50)), split_name="pretrain")
    logging.info(f"BVP samples: {len(ds)}")

    sim = PhysioNoiseSimulator(fs=fs, noise_probs={
        'motion': 0.9, 'illumination': 0.8, 'sensor': 1.0,
        'compression': 0.6, 'emi': 0.5, 'dropout': 0.4,
    })
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=get_num_workers(TRAIN_CONFIG.get("NUM_WORKERS", 2)),
                        pin_memory=True, drop_last=True, collate_fn=DenoisingCollator(sim))

    logging.info(f"Batches/epoch: {len(loader)} | Noise: motion+illum+sensor+comp+emi+dropout")

    model = RemoteSSM(d_model=d_model, seq_len=seq_len, n_ssm_blocks=n_ssm_blocks,
                      roi_size=TRAIN_CONFIG.get("FACE_ROI_SIZE", 32),
                      dropout=TRAIN_CONFIG.get("DROPOUT", 0.15), fs=fs,
                      use_fdf=True).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Params: {n_params:.2f}M | Arch: V11 + COB init")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    amp = bool(TRAIN_CONFIG.get("AMP", True) and device.type == "cuda")
    scaler = create_grad_scaler(amp)
    ctx = get_autocast_context(device, amp)

    # Resume
    start_epoch, best_loss = 0, float("inf")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        logging.info(f"Resumed at epoch {start_epoch}")

    w_wave, w_pearson, w_freq = 1.0, 0.5, 0.3

    for epoch in range(start_epoch, n_epochs):
        model.train()
        losses = []
        for b, batch in enumerate(loader):
            noisy, clean = batch['noisy'].to(device), batch['clean'].to(device)
            optimizer.zero_grad()
            with ctx:
                pred = model(noisy_bvp=noisy, mode='denoise')
                loss = (w_wave * F.l1_loss(pred, clean) +
                        w_pearson * pearson_loss(pred, clean) +
                        w_freq * cardiac_freq_loss(pred, clean, fs))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn = torch.nn.utils
            nn.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if b % 50 == 0:
                logging.info(f"Epoch {epoch+1}/{n_epochs} | B {b}/{len(loader)} | Loss {loss.item():.6f}")

        scheduler.step()
        avg = np.mean(losses)
        logging.info(f"Epoch {epoch+1}/{n_epochs} | Avg Loss: {avg:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            logging.info(f"  ✓ Best (loss={best_loss:.6f})")
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "best_loss": best_loss}, resume_path)

    logging.info(f"Stage 1 done. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
