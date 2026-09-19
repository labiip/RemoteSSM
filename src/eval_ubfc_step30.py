"""UBFC 消融 step30 统一复核: full / no_residual / no_cob / no_outband / scalar-hr.

全部用同一协议: 32px cardio_v5_bvphr, test 40-49, VAL_STEP=30 (481 窗),
FFT(BVP) soft-argmax 读 HR, BVP 衍生标签. 与论文表同口径.
用法: python3 -u eval_ubfc_step30.py
"""
import os
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 32
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v5_bvphr"
config.TRAIN_CONFIG["VAL_STEP"] = 30

from evaluate import evaluate

UBFC_TEST = list(range(40, 50))

MODELS = [
    ("full (COB+residual)", "ubfc_pure", "remotessm_ubfc_pure_best.pth"),
    ("w/o residual", "ubfc_ablate_no_residual", "remotessm_ubfc_ablate_no_residual_best.pth"),
    ("w/o COB (RANDOM_A)", "ubfc_ablate_no_cob", "remotessm_ubfc_ablate_no_cob_best.pth"),
    ("w/o out-of-band", "ubfc_ablate_no_outband", "remotessm_ubfc_ablate_no_outband_best.pth"),
    ("scalar HR supervision", "ubfc_ablate_bpm_supervision", "remotessm_ubfc_ablate_bpm_supervision_best.pth"),
]

if __name__ == "__main__":
    for name, d, fn in MODELS:
        mp = os.path.join(PROJ, "outputs", d, fn)
        if not os.path.exists(mp):
            print(f"[SKIP] missing {mp}")
            continue
        print(f"\n===== {name} =====")
        evaluate(model_path=mp, test_subjects=UBFC_TEST)
