"""OOD 快速验证: 静态训练模型 (full / w/o residual / w/o COB) 直接测 MMPD 全部 33 subject.

leave-domain 场景下检验残差连接与 COB 的"保底/先验"价值:
  full ubfc_pure 已知 leave_mmpd = 20.76 (step30, 23651 窗)
  若 w/o residual / w/o COB 在 OOD 明显更差 -> 两机制具有跨域保底价值 (有意义的对比)
用法: python3 -u eval_leave_step30.py
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

MMPD_ALL = list(range(51, 84))

MODELS = [
    ("full (COB+residual)", "ubfc_pure", "remotessm_ubfc_pure_best.pth"),
    ("w/o residual", "ubfc_ablate_no_residual", "remotessm_ubfc_ablate_no_residual_best.pth"),
    ("w/o COB (RANDOM_A)", "ubfc_ablate_no_cob", "remotessm_ubfc_ablate_no_cob_best.pth"),
]

if __name__ == "__main__":
    for name, d, fn in MODELS:
        mp = os.path.join(PROJ, "outputs", d, fn)
        if not os.path.exists(mp):
            print(f"[SKIP] missing {mp}")
            continue
        print(f"\n===== LEAVE-OOD {name} =====")
        evaluate(model_path=mp, test_subjects=MMPD_ALL)
