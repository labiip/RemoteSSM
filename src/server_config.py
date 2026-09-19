"""Server-side configuration for RemoteSSM training."""
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

TRAIN_CONFIG = {
    "DATASET_ROOT": os.path.expanduser("~/rppg/datasets/UBFC"),
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-3,
    "EPOCHS": 300,
    "TRAIN_STEP": 10,
    "VAL_STEP": 300,
    "NUM_WORKERS": 4,
    "AMP": True,
    "GRAD_CLIP": 1.0,
    "WAVE_LOSS_W": 2.0,
    "BPM_LOSS_W": 3.0,
    "FREQ_LOSS_W": 1.5,
    "HR_HEAD_LOSS_W": 3.0,
    "CURRICULUM_SEQ_LENS": [100, 200, 300],
    "CURRICULUM_EPOCHS": [80, 160, 300],
    "TRAIN_SUBJECTS": tuple(range(1, 35)),
    "VAL_SUBJECTS": tuple(range(35, 40)),
    "TEST_SUBJECTS": tuple(range(40, 50)),
    "ADV_LOSS_W": 0.05,
    "GRL_WARMUP_EPOCHS": 80,
    "GRL_ALPHA_MAX": 0.3,
    "DROPOUT": 0.35,
    "INPUT_AUG_SCALE": 0.10,
    "INPUT_AUG_NOISE": 0.01,
    "FACE_AUG_BRIGHTNESS": 0.2,
    "FACE_AUG_CONTRAST": 0.2,
    "BVP_SPECTRAL_AUG": True,
    "BVP_SPECTRAL_NOISE": 0.08,
    "ENABLE_TIME_ALIGNMENT": True,
    "USE_BVP_DERIVED_HR_LABEL": True,
    "ENABLE_QUALITY_FILTER": False,   # Off by default for UBFC
    "MIN_VALID_HR_RATIO": 0.60,
    "MIN_BVP_BAND_ENERGY_RATIO": 0.20,
    "MIN_BVP_PEAK_RATIO": 0.12,
    "MIN_RGB_STD": 0.30,
    "FACE_ROI_SIZE": 32,
    "CACHE_VERSION": "realssm_v3",
}

# Pretrain (Stage 1) config
PRETRAIN_CONFIG = {
    "BATCH_SIZE": 8,
    "EPOCHS": 100,
    "LEARNING_RATE": 2e-4,
    "WEIGHT_DECAY": 1e-4,
    "AMP": True,
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
    "KALMAN_Q_HR": 0.02,
    "KALMAN_R_HR": 0.05,
    "HR_RANGE": (40, 180),
}
