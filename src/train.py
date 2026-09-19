"""
Stage 2: HR 微调 (RemoteSSM, 申报书算法 + COB预训练)

加载 Stage 1 预训练权重 → 在 UBFC 上端到端 HR 微调

训练策略 (V11 proven pipeline):
  - 课程学习: Len 100→200→300 @ [80, 160, 300]
  - GRL 受试者对抗: warmup=80, alpha=0.3
  - BVP 衍生 HR 标签 (文献标准, 和 SOTA 可比)
  - 多损失联合: Pearson + Freq + BPM + HR Head
  - 数据增强: face brightness/contrast + signal scale/noise + BVP spectral
  - 双路输入: Face ROI + BGR 信号 → GatedFusion

评估:
  - 验证集 MAE 选最优
  - 测试集 MAE/RMSE/Pearson/Acc

非静态→静态: trainer 支持自定义 train/val/test subject_ids,
  通过 config.py 调整即可实现跨域实验
"""

import os
import math
import random
import logging
import platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from contextlib import nullcontext

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from dataset import TitanrPPGDataset
from model import RemoteSSM, SubjectDiscriminator, grad_reverse
from utils import DifferentiablePhysioExtractor


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def get_num_workers(requested):
    if requested is None or requested <= 0: return 0
    safe = min(int(requested), os.cpu_count() or 1)
    return min(safe, 4) if platform.system().lower() == "windows" else safe


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_autocast_context(device, enabled):
    if not enabled: return nullcontext()
    dt = device.type if hasattr(device, 'type') else str(device)
    try:
        return torch.amp.autocast(device_type=dt, enabled=True)
    except Exception:
        if dt == "cuda":
            try: return torch.cuda.amp.autocast(enabled=True)
            except: pass
    return nullcontext()


def create_grad_scaler(enabled):
    try: return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        class D:  # noqa
            def scale(self, l): return l
            def unscale_(self, o): pass
            def step(self, o): o.step()
            def update(self): pass
        return D()


# ═══════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════

def pearson_loss(pred, target):
    pred, target = pred.view(pred.shape[0], -1), target.view(target.shape[0], -1)
    # fp32 computation for numerical safety
    pred_f, target_f = pred.float(), target.float()
    pm, tm = pred_f.mean(1, keepdim=True), target_f.mean(1, keepdim=True)
    pd, td = pred_f - pm, target_f - tm
    num = (pd * td).sum(1)
    den = torch.sqrt((pd**2).sum(1) + 1e-6) * torch.sqrt((td**2).sum(1) + 1e-6)
    return (1.0 - (num / den).mean()).to(pred.dtype)


def masked_frequency_loss(pred, target, fs=30.0, fmin=0.7, fmax=3.0):
    """L1 on normalized cardiac-band magnitude."""
    orig = pred.dtype
    if orig != torch.float32: pred, target = pred.float(), target.float()
    pf = torch.abs(torch.fft.rfft(pred, dim=1))
    tf = torch.abs(torch.fft.rfft(target, dim=1))
    n = pred.shape[1]
    freqs = torch.fft.rfftfreq(n, d=1.0/fs).to(pred.device)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.any():
        pb = pf[:, mask] / (pf[:, mask].sum(1, keepdim=True) + 1e-8)
        tb = tf[:, mask] / (tf[:, mask].sum(1, keepdim=True) + 1e-8)
        return F.l1_loss(pb, tb)
    return torch.tensor(0.0, device=pred.device)


def out_of_band_energy_loss(pred, fs=30.0, fmin=0.7, fmax=3.0):
    """物理约束 (PINN 式): 输出频谱能量应集中在心率带内, 惩罚带外能量占比.

    生理依据: 脉搏信号的主要能量集中在心率频带 (0.7-3.0 Hz), 带外能量
    (运动伪影/噪声) 应被抑制. 作为正则化项辅助 FREQ loss.
    """
    orig = pred.dtype
    if orig != torch.float32:
        pred = pred.float()
    pf = torch.abs(torch.fft.rfft(pred, dim=1))
    n = pred.shape[1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs).to(pred.device)
    mask = (freqs >= fmin) & (freqs <= fmax)
    total = pf.sum(1) + 1e-8
    band = pf[:, mask].sum(1)
    return (1.0 - band / total).mean()


def robust_bpm_loss(pred_hr, target_hr, beta=3.0):
    return F.smooth_l1_loss(pred_hr.view(-1), target_hr.view(-1), beta=beta)


# ═══════════════════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════════════════

def input_augmentation(x_real, x_imag, scale=0.10, noise_std=0.01):
    B, device = x_real.shape[0], x_real.device
    s = 1.0 + (torch.rand(B, 1, 1, device=device) * 2 - 1) * scale
    return (x_real * s + torch.randn_like(x_real) * noise_std,
            x_imag * s + torch.randn_like(x_imag) * noise_std)


def face_roi_augmentation(face_roi, brightness=0.2, contrast=0.2, noise=0.06):
    B, T, H, W, C = face_roi.shape
    d = face_roi.device
    a = face_roi.clone()
    if brightness > 0:
        a = a + (torch.rand(B, 1, 1, 1, 1, device=d) * 2 - 1) * brightness
    if contrast > 0:
        m = a.mean(dim=(-3, -2, -1), keepdim=True)
        a = m + (1.0 + (torch.rand(B, 1, 1, 1, 1, device=d) * 2 - 1) * contrast) * (a - m)
    if noise > 0:
        a = a + torch.randn_like(a) * noise
    return torch.clamp(a, 0.0, 1.0)


def bvp_spectral_augmentation(target_bvp, fs=30.0, noise_std=0.08, hr_band=(0.7, 3.0)):
    B, L = target_bvp.shape
    d = target_bvp.device
    tf = torch.fft.rfft(target_bvp.float(), dim=1)
    nf = tf.shape[1]
    freqs = torch.fft.rfftfreq(L, d=1.0/fs).to(d)
    nm = ((freqs >= 0.1) & (freqs <= hr_band[0])) | ((freqs >= hr_band[1]) & (freqs <= 10.0))
    if nm.any():
        n = torch.randn(B, nf, device=d) * noise_std * nm.float().unsqueeze(0)
        tf = tf + n
    return torch.fft.irfft(tf, n=L, dim=1).to(target_bvp.dtype)


# ═══════════════════════════════════════════════════════════════
# Curriculum
# ═══════════════════════════════════════════════════════════════

def get_curriculum_seq_len(epoch, seq_lens, epoch_boundaries):
    if seq_lens is None or epoch_boundaries is None: return None
    for cl, ce in zip(seq_lens, epoch_boundaries):
        if epoch <= ce: return cl
    return seq_lens[-1]


# ═══════════════════════════════════════════════════════════════
# Pretrained weight loading
# ═══════════════════════════════════════════════════════════════

def load_pretrained_backbone(model, ckpt_path, device):
    """Load Stage 1 pretrained weights (SSM blocks + FDF + BVP head)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    ms = model.state_dict()
    prefix_list = ["ssm_blocks", "final_fdf", "output_norm", "output_dropout", "fc_out"]
    loaded = 0
    for k in sd:
        if any(k.startswith(p) for p in prefix_list):
            if k in ms and sd[k].shape == ms[k].shape:
                ms[k] = sd[k]; loaded += 1
    model.load_state_dict(ms)
    logging.info(f"Pretrained: {loaded} parameters (FaceStem/TemporalStem/HR-Head re-init)")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def train_model(train_subjects=None, val_subjects=None, out_dir=None, tag=None):
    set_seed(COMMON_CONFIG.get("SEED", 42))
    project_dir = out_dir if out_dir else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(project_dir, exist_ok=True)

    base = f"remotessm_{tag}" if tag else "remotessm"
    log_path = os.path.join(project_dir, f"{base}_train.log")
    best_path = os.path.join(project_dir, f"{base}_best.pth")
    final_path = os.path.join(project_dir, f"{base}_final.pth")
    resume_path = os.path.join(project_dir, f"{base}_resume.pth")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"RemoteSSM Finetuning | Device: {device}")

    # ── Config ──
    seq_len = COMMON_CONFIG["SEQ_LEN"]
    d_model = COMMON_CONFIG["D_MODEL"]
    n_ssm_blocks = COMMON_CONFIG.get("N_SSM_BLOCKS", 2)
    fs = COMMON_CONFIG["FS"]
    dataset_root = TRAIN_CONFIG["DATASET_ROOT"]
    batch_size = TRAIN_CONFIG["BATCH_SIZE"]
    lr = TRAIN_CONFIG["LEARNING_RATE"]
    weight_decay = TRAIN_CONFIG["WEIGHT_DECAY"]
    epochs = TRAIN_CONFIG["EPOCHS"]
    train_step = TRAIN_CONFIG["TRAIN_STEP"]
    val_step = TRAIN_CONFIG["VAL_STEP"]
    num_workers = get_num_workers(TRAIN_CONFIG.get("NUM_WORKERS", 4))
    use_amp = TRAIN_CONFIG["AMP"]
    grad_clip = TRAIN_CONFIG.get("GRAD_CLIP", 1.0)
    dropout = TRAIN_CONFIG.get("DROPOUT", 0.15)
    roi_size = TRAIN_CONFIG.get("FACE_ROI_SIZE", 32)

    # Subjects (可自定义以实现非静态→静态)
    if train_subjects is None:
        train_subjects = TRAIN_CONFIG["TRAIN_SUBJECTS"]
    if val_subjects is None:
        val_subjects = TRAIN_CONFIG["VAL_SUBJECTS"]

    # Adversarial
    adv_loss_w = TRAIN_CONFIG.get("ADV_LOSS_W", 0.05)
    grl_warmup = TRAIN_CONFIG.get("GRL_WARMUP_EPOCHS", 80)
    grl_alpha_max = TRAIN_CONFIG.get("GRL_ALPHA_MAX", 0.3)

    # Augmentation
    ia_s = TRAIN_CONFIG.get("INPUT_AUG_SCALE", 0.10)
    ia_n = TRAIN_CONFIG.get("INPUT_AUG_NOISE", 0.01)
    fa_b = TRAIN_CONFIG.get("FACE_AUG_BRIGHTNESS", 0.2)
    fa_c = TRAIN_CONFIG.get("FACE_AUG_CONTRAST", 0.2)
    bvp_aug = TRAIN_CONFIG.get("BVP_SPECTRAL_AUG", True)
    bvp_noise = TRAIN_CONFIG.get("BVP_SPECTRAL_NOISE", 0.08)
    outband_w = TRAIN_CONFIG.get("OUTBAND_LOSS_W", 0.3)   # 带外能量物理约束权重
    lr_warmup = TRAIN_CONFIG.get("LR_WARMUP_EPOCHS", 5)    # warmup epoch 数
    accum_steps = max(1, int(TRAIN_CONFIG.get("GRAD_ACCUM", 1)))  # 梯度累积: 等效放大 batch 不增显存

    # Curriculum
    curr_lens = TRAIN_CONFIG.get("CURRICULUM_SEQ_LENS", [100, 200, 300])
    curr_eps = TRAIN_CONFIG.get("CURRICULUM_EPOCHS", [80, 160, 300])

    # HR labels
    use_bvp_hr = TRAIN_CONFIG.get("USE_BVP_DERIVED_HR_LABEL", True)

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    # ── Datasets ──
    train_ds = TitanrPPGDataset(root_dir=dataset_root, seq_len=seq_len, step=train_step,
                                subject_ids=train_subjects, split_name="train")
    val_ds = TitanrPPGDataset(root_dir=dataset_root, seq_len=seq_len, step=val_step,
                              subject_ids=val_subjects, split_name="val")

    if len(train_ds) == 0: raise RuntimeError("Empty train dataset!")
    if len(val_ds) == 0: raise RuntimeError(f"Empty val dataset (subjects={val_subjects})!")

    logging.info(f"Train: {len(train_ds)} samples, subjects {train_subjects}")
    logging.info(f"Val:   {len(val_ds)} samples, subjects {val_subjects}")
    logging.info(f"HR labels: {'BVP-derived (SOTA-comparable)' if use_bvp_hr else 'ECG gold-standard'}")
    logging.info(f"d={d_model} blocks={n_ssm_blocks} B={batch_size} "
                 f"dropout={dropout} wd={weight_decay}")
    # V11-style: random SSM init + curriculum is the proven pipeline
    logging.info(f"V11: random init + curriculum ({curr_lens} @ {curr_eps})")
    logging.info(f"GRL: warmup={grl_warmup} adv_w={adv_loss_w} alpha_max={grl_alpha_max}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # ── Model ──
    model = RemoteSSM(d_model=d_model, seq_len=seq_len, n_ssm_blocks=n_ssm_blocks,
                      roi_size=roi_size, dropout=dropout, fs=fs, use_fdf=True).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Params: {n_params:.2f}M (V11 random init)")

    # PRETRAIN_FULL: 从已有 checkpoint 加载完整权重作初始化 (迁移微调用, 含 FrameStem)
    pretrain_full = os.environ.get("PRETRAIN_FULL", "")
    if pretrain_full:
        if os.path.exists(pretrain_full):
            ck = torch.load(pretrain_full, map_location=device)
            sd = ck.get("model_state_dict", ck)
            miss, unexp = model.load_state_dict(sd, strict=False)
            logging.info(f"PRETRAIN_FULL: {os.path.basename(pretrain_full)} | missing={len(miss)} unexpected={len(unexp)}")
            # RANDOM_A 消融 (配合预训练微调): 预训练权重会覆盖构造期的随机 A,
            # 故加载后重新随机化 A, 使该消融只去掉 COB 频率先验, 其余权重与主行完全一致.
            if os.environ.get("RANDOM_A", "0") == "1":
                for _blk in model.ssm_blocks:
                    _blk._init_A_params()
                logging.info("PRETRAIN_FULL + RANDOM_A=1: A re-initialized randomly (COB prior removed)")
        else:
            logging.warning(f"PRETRAIN_FULL path not found: {pretrain_full}")

    # torch.compile (skip on Windows - no Triton)
    if os.name != 'nt':
        try:
            if hasattr(torch, 'compile') and device.type == "cuda":
                cc = torch.cuda.get_device_capability(0)
                if cc[0] >= 7 and torch.cuda.is_available():
                    model = torch.compile(model, mode="reduce-overhead")
                    logging.info(f"torch.compile enabled (CC={cc[0]}.{cc[1]})")
        except Exception:
            logging.info("torch.compile skipped")

    # Discriminator
    num_subj = len(train_subjects)
    discriminator = SubjectDiscriminator(d_model=d_model, num_subjects=num_subj).to(device)

    # Optimizers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    disc_optimizer = optim.AdamW(discriminator.parameters(), lr=lr * 0.5, weight_decay=weight_decay)
    # warmup + cosine decay (小数据上提升泛化: 初期小步升温避免震荡, 后期余弦衰减)
    def _lr_lambda(ep):
        if ep < lr_warmup:
            return (ep + 1) / max(1, lr_warmup)
        prog = (ep - lr_warmup) / max(1, epochs - lr_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    disc_scheduler = LambdaLR(disc_optimizer, lr_lambda=_lr_lambda)

    amp_ok = bool(use_amp and device.type == "cuda")
    scaler = create_grad_scaler(amp_ok)
    disc_scaler = create_grad_scaler(amp_ok)

    extractor = DifferentiablePhysioExtractor(fs=fs).to(device)

    # Loss weights
    w_wave = TRAIN_CONFIG.get("WAVE_LOSS_W", 2.0)
    w_bpm = TRAIN_CONFIG.get("BPM_LOSS_W", 3.0)
    w_freq = TRAIN_CONFIG.get("FREQ_LOSS_W", 1.5)
    w_hh = TRAIN_CONFIG.get("HR_HEAD_LOSS_W", 3.0)

    # Resume
    best_val_mae = float("inf")
    start_epoch = 1
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
        # scheduler 类型可能变化 (Cosine→Lambda warmup+cosine), 失败则重新从头调度
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            logging.warning("scheduler state 不兼容, 重新初始化 LR 调度")
        try:
            disc_scheduler.load_state_dict(ckpt["disc_scheduler_state_dict"])
        except Exception:
            logging.warning("disc_scheduler state 不兼容, 重新初始化 LR 调度")
        if "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "disc_scaler_state_dict" in ckpt: disc_scaler.load_state_dict(ckpt["disc_scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt.get("best_val_mae", float("inf"))
        logging.info(f"Resumed at epoch {start_epoch} (best={best_val_mae:.2f})")

    # ── Training ──
    autocast_ctx = get_autocast_context(device, amp_ok)
    nan_streak = 0

    # Pre-training diagnostic: verify forward pass produces finite values
    with torch.no_grad():
        diag_batch = next(iter(train_loader))
        df = diag_batch['face_roi'][:2].to(device)
        dxr = diag_batch['input_real'][:2].to(device)
        dxi = diag_batch['input_imag'][:2].to(device)
        model.eval()
        dbvp, dhr = model(face_roi=df, x_real=dxr, x_imag=dxi, mode='finetune')
        model.train()
        dbvp_rng = f"[{dbvp.min().item():.3f}, {dbvp.max().item():.3f}]"
        dhr_rng = f"[{dhr.min().item():.3f}, {dhr.max().item():.3f}]"
        has_nan_diag = torch.isnan(dbvp).any() or torch.isnan(dhr).any()
        logging.info(f"Pre-flight check: BVP {dbvp_rng}, HR {dhr_rng}, NaN={has_nan_diag}")
        if has_nan_diag:
            logging.error("Model produces NaN on first forward pass — aborting.")
            return

    for epoch in range(start_epoch, epochs + 1):
        cur_seq_len = get_curriculum_seq_len(epoch, curr_lens, curr_eps)
        model.train(); discriminator.train()

        grl_alpha = grl_alpha_max * min(1.0, (epoch - grl_warmup) / (epochs * 0.3)) if epoch > grl_warmup else 0.0

        ep_adv, ep_loss, nb = 0.0, 0.0, 0
        optimizer.zero_grad(); disc_optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            face = batch['face_roi'].to(device)
            xr = batch['input_real'].to(device)
            xi = batch['input_imag'].to(device)
            t_bvp = batch['target_bvp'].to(device)
            t_hr = batch['target_hr'].to(device)
            subj = batch['subject_id'].to(device)

            if cur_seq_len:
                face = face[:, :cur_seq_len]; xr = xr[:, :cur_seq_len]
                xi = xi[:, :cur_seq_len]; t_bvp = t_bvp[:, :cur_seq_len]

            face_a = face_roi_augmentation(face, fa_b, fa_c, 0.06)
            xr_a, xi_a = input_augmentation(xr, xi, ia_s, ia_n)
            # FACE_ONLY 模式: 信号支路置零, 模型纯靠 face_roi 支路学习
            # 用于验证 MMPD 上空间信息是否可学 (诊断用, 非正式实验)
            if os.environ.get("FACE_ONLY", "0") == "1":
                xr_a = torch.zeros_like(xr_a)
                xi_a = torch.zeros_like(xi_a)
            t_bvp_a = bvp_spectral_augmentation(t_bvp, fs, bvp_noise) if bvp_aug else t_bvp

            with autocast_ctx:
                bvp_p, hr_p, feat = model(face_roi=face_a, x_real=xr_a, x_imag=xi_a,
                                          mode='finetune', return_features=True)

                loss_wave = pearson_loss(bvp_p, t_bvp_a)
                loss_freq = masked_frequency_loss(bvp_p, t_bvp_a, fs=fs)
                # BPM 数值监督温度: 默认 0.05; 消融 bpm_supervision 用 0.5 (梯度饱和阈值, 见项目经验)
                hr_fft = extractor(torch.clamp(bvp_p, -3.0, 3.0),
                                   freq_range=PHYSIO_CONFIG["HR_BAND"],
                                   temperature=float(os.environ.get("BPM_TEMP", "0.05")))
                loss_bpm = F.mse_loss(hr_fft, t_hr)
                loss_head = robust_bpm_loss(hr_p, t_hr)

                if grl_alpha > 0:
                    adv_f = grad_reverse(feat, alpha=grl_alpha)
                    loss_adv = F.cross_entropy(discriminator(adv_f), subj)
                    ep_adv += (discriminator(feat).argmax(1) == subj).float().mean().item()
                else:
                    loss_adv = torch.tensor(0.0, device=device)

                total = (w_wave * loss_wave + w_bpm * loss_bpm +
                         w_freq * loss_freq + w_hh * loss_head +
                         outband_w * out_of_band_energy_loss(bvp_p, fs=fs) +
                         adv_loss_w * loss_adv)

            # ── NaN保护: 跳过本微批, 不污染已累积梯度 ──
            if torch.isnan(total) or torch.isinf(total):
                nan_streak += 1
                logging.warning(f"Epoch {epoch} B {nb}: NaN loss (streak={nan_streak})")
                if nan_streak > 50:
                    logging.error("Persistent NaN, stopping.")
                    return
                continue

            scaler.scale(total).backward()
            ep_loss += total.item(); nb += 1

            # 梯度累积: 每 accum_steps 个微批才 unscale/clip/step 一次
            # (等效放大 batch size 而不增加显存; BN 统计仍按微批更新, 属标准行为)
            do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if not do_step:
                continue

            scaler.unscale_(optimizer)
            has_nan = False
            nan_params = []
            for name, p in model.named_parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan = True
                    nan_params.append(name)
            if has_nan:
                nan_streak += 1
                if nan_streak <= 3:
                    logging.warning(f"Epoch {epoch} B {nb}: NaN grad (streak={nan_streak}) in: {nan_params[:5]}")
                else:
                    logging.warning(f"Epoch {epoch} B {nb}: NaN grad (streak={nan_streak})")
                if nan_streak > 50:
                    logging.error("Persistent NaN grad, stopping.")
                    return
                optimizer.zero_grad(); disc_optimizer.zero_grad()   # 丢弃整个累积周期
                scaler.update()
                continue
            nan_streak = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad()

            if grl_alpha > 0:
                disc_optimizer.zero_grad()
                with torch.no_grad(): df = feat.detach()
                with autocast_ctx:
                    d_loss = F.cross_entropy(discriminator(df), subj)
                disc_scaler.scale(d_loss).backward()
                disc_scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                disc_scaler.step(disc_optimizer); disc_scaler.update()

        avg_loss = ep_loss / max(nb, 1)
        avg_acc = ep_adv / max(nb, 1)
        scheduler.step(); disc_scheduler.step()

        # ── Validation ──
        model.eval()
        val_errs = []
        val_hp = []   # predicted HR via FFT extractor (paper metric)
        val_hh = []   # predicted HR via HR head
        val_ht = []   # target HR
        with torch.no_grad():
            for batch in val_loader:
                f = batch['face_roi'].to(device)
                xr = batch['input_real'].to(device)
                xi = batch['input_imag'].to(device)
                thr = batch['target_hr'].to(device)
                with autocast_ctx:
                    bp, hr_head = model(face_roi=f, x_real=xr, x_imag=xi, mode='finetune')
                    hp = extractor(torch.clamp(bp, -3.0, 3.0),
                                   freq_range=PHYSIO_CONFIG["HR_BAND"], temperature=0.05)
                    val_errs.extend((hp - thr).abs().cpu().tolist())
                    val_hp.extend(hp.cpu().tolist())
                    val_hh.extend(hr_head.cpu().tolist())
                    val_ht.extend(thr.cpu().tolist())

        def _pearson(a, b):
            a = np.asarray(a, np.float32).flatten()
            b = np.asarray(b, np.float32).flatten()
            if len(a) < 2 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        val_mae = np.mean(val_errs) if val_errs else 999.0
        val_hr_pearson = _pearson(val_hp, val_ht)
        val_hh_mae = float(np.mean(np.abs(np.asarray(val_hh, np.float32).flatten() -
                                          np.asarray(val_ht, np.float32).flatten()))) if val_hh else 999.0
        val_hh_pearson = _pearson(val_hh, val_ht)

        log_msg = (f"Epoch {epoch:3d}/{epochs} | Len={cur_seq_len or seq_len} | Loss {avg_loss:.4f} "
                   f"| Val FFT {val_mae:.2f} | Val r(HR) {val_hr_pearson:.3f} "
                   f"| HH MAE {val_hh_mae:.2f} | HH r {val_hh_pearson:.3f}")
        if grl_alpha > 0: log_msg += f" | AdvAcc={avg_acc:.1%}"
        logging.info(log_msg)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            logging.info(f"  ✓ Best: {best_val_mae:.2f} BPM")

        torch.save({"model_state_dict": model.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "disc_scheduler_state_dict": disc_scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "disc_scaler_state_dict": disc_scaler.state_dict(),
                    "epoch": epoch, "best_val_mae": best_val_mae,
                    "cur_seq_len": cur_seq_len}, resume_path)

    torch.save({"model_state_dict": model.state_dict()}, final_path)
    logging.info(f"Done. Best Val FFT: {best_val_mae:.2f} BPM")


if __name__ == "__main__":
    train_model()
