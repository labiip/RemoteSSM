"""RemoteSSM 多数据集联合训练配置 (对标 SCI 二区).

三数据集: UBFC(49) + MMPD(33) + PURE(10) ≈ 92 受试者联合训练.
全局 subject id 命名空间 (不重叠):
  UBFC: 1-49
  MMPD: 51-83  (offset +50)
  PURE: 101-110 (offset +100)
"""
from pathlib import Path
import os, torch

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.path.expanduser("~/rppg/output"))

COMMON_CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "FS": 30.0,
    "SEQ_LEN": 300,
    "D_MODEL": 256,
    "N_SSM_BLOCKS": 2,
    "SEED": 42
}

# 数据集 -> 全局 subject id 映射 (供跨域实验使用)
DATASET_SUBJECTS = {
    "UBFC": list(range(1, 50)),
    "MMPD": list(range(51, 84)),
    "PURE": list(range(101, 111)),
}

# 主实验: 三数据集联合训练 subject-independent 划分 (train 62 / val 12 / test 18)
TRAIN_SUBJECTS = tuple(
    list(range(1, 35))
)
VAL_SUBJECTS = tuple(
    list(range(35, 40))
)
TEST_SUBJECTS = tuple(
    list(range(40, 50))
)

TRAIN_CONFIG = {
    "DATASET_ROOT": os.path.expanduser("~/rppg/datasets/combined"),
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 5e-3,
    "EPOCHS": 300,
    "TRAIN_STEP": 10,
    "VAL_STEP": 300,
    "NUM_WORKERS": 4,
    "AMP": False,
    "GRAD_CLIP": 0.5,
    "WAVE_LOSS_W": 2.0,       # negPearson 波形监督 (SOTA 标准, 主导)
    "BPM_LOSS_W": 0.0,        # 数值HR监督已删除: softmax-argmax 温度口径与 FFT 峰值不一致, 造成评估假象
    "FREQ_LOSS_W": 3.0,       # 频谱幅度监督 (约束输出峰值对齐 BVP 峰值)
    "HR_HEAD_LOSS_W": 0.0,    # HR直接回归 (辅助, 已关)
    "CURRICULUM_SEQ_LENS": [300],
    "CURRICULUM_EPOCHS": [300],
    "TRAIN_SUBJECTS": TRAIN_SUBJECTS,
    "VAL_SUBJECTS": VAL_SUBJECTS,
    "TEST_SUBJECTS": TEST_SUBJECTS,
    # GRL 已禁用 (MMPD 小数据 22 subject 下会反向删心率信息, 30 epoch 后激活有害)
    "ADV_LOSS_W": 0.0,
    "GRL_WARMUP_EPOCHS": 30,
    "GRL_ALPHA_MAX": 0.0,
    "DROPOUT": 0.40,
    "INPUT_AUG_SCALE": 0.10,
    "INPUT_AUG_NOISE": 0.01,
    "FACE_AUG_BRIGHTNESS": 0.4,
    "FACE_AUG_CONTRAST": 0.4,
    "BVP_SPECTRAL_AUG": True,
    "BVP_SPECTRAL_NOISE": 0.08,
    "OUTBAND_LOSS_W": 0.3,       # 带外能量物理约束 (PINN 式正则, 输出频谱应集中心率带)
    "LR_WARMUP_EPOCHS": 5,       # warmup epoch 数 (小数据提升泛化)
    "ENABLE_TIME_ALIGNMENT": False,
    "USE_BVP_DERIVED_HR_LABEL": True,   # 官方HR列中位数 (消~3.25BPM FFT量化噪声)
    "ENABLE_QUALITY_FILTER": True,
    "MIN_VALID_HR_RATIO": 0.60,
    "MIN_BVP_BAND_ENERGY_RATIO": 0.20,
    "MIN_BVP_PEAK_RATIO": 0.12,
    "MIN_RGB_STD": 0.30,
    "FACE_ROI_SIZE": 32,
    "CACHE_VERSION": "cardio_v5_bvphr",       # v4: 官方HR标签 + 多数据集
}

# Pretrain (Stage 1) config
PRETRAIN_CONFIG = {
    "BATCH_SIZE": 8,
    "EPOCHS": 100,
    "LEARNING_RATE": 2e-4,
    "WEIGHT_DECAY": 1e-4,
    "AMP": False,
    "GRAD_CLIP": 1.0,
    "BVP_CACHE_DIR": os.path.expanduser("~/rppg/output/bvp_cache"),
    "PRETRAIN_SAVE_PATH": str(OUTPUT_DIR / "pretrain_stage1_v3.pth"),
    "NOISE_PROBS": {
        "motion": 0.9,
        "illumination": 0.8,
        "sensor": 1.0,
        "compression": 0.6,
        "emi": 0.5,
        "dropout": 0.4,
    },
}

INFER_CONFIG = {
    "MODEL_PATH": str(OUTPUT_DIR / "remotessm_best.pth"),
    "MAX_FACE_LOST_FRAMES": 30,
    "FACE_TRACKING_SMOOTH": 0.15,
    "FACE_DETECT_INTERVAL": 2,
    "ENABLE_SKIN_MASK": True,
    "ENABLE_DEBUG_OVERLAY": False,
    "KALMAN_Q_HR": 0.02,
    "KALMAN_R_HR": 0.05,
    "HR_RANGE": (40, 180),
}

PHYSIO_CONFIG = {
    "HR_BAND": (0.7, 3.0),
    "RR_BAND": (0.1, 0.5),
}
