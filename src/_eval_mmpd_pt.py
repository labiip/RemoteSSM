"""评估 MMPD 预训练微调的当前最优 checkpoint (test 78-83)."""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"

from evaluate import evaluate

evaluate(model_path=os.path.join(PROJ, "outputs", "mmpd_facemain_pt", "remotessm_mmpd_facemain_pt_best.pth"),
         test_subjects=list(range(78, 84)))
