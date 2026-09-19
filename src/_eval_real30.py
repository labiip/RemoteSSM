"""实数 SSM 模型 step30 统一口径评估 (与双路 13.55 同协议对比)."""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
config.TRAIN_CONFIG["VAL_STEP"] = 30

from evaluate import evaluate

evaluate(model_path=os.path.join(PROJ, "outputs", "mmpd_real_pt", "remotessm_mmpd_real_pt_best.pth"),
         test_subjects=list(range(78, 84)))
