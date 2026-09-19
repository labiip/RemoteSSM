"""4 个 UBFC 消融 checkpoint 统一 step30 重评 (与主表同协议)."""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
from evaluate import evaluate

config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 32
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v5_bvphr"
config.TRAIN_CONFIG["VAL_STEP"] = 30

TEST = list(range(40, 50))
NAMES = ["no_residual", "bpm_supervision", "no_cob", "no_outband"]

for name in NAMES:
    print(f"\n{'='*56}\nABLATE: {name} (step30)\n{'='*56}")
    evaluate(model_path=os.path.join(PROJ, "outputs", f"ubfc_ablate_{name}",
                                     f"remotessm_ubfc_ablate_{name}_best.pth"),
             test_subjects=TEST)
