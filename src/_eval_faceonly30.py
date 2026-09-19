"""face-only 复数模型 step30 统一口径评估 (MMPD 三向对比同协议)."""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
config.TRAIN_CONFIG["VAL_STEP"] = 30

from evaluate import evaluate

evaluate(model_path=os.path.join(PROJ, "outputs", "mmpd_facemain_pt", "remotessm_mmpd_facemain_pt_best.pth"),
         test_subjects=list(range(78, 84)))
