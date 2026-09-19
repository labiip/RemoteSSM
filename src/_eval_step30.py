"""统一口径 (step=30 重叠窗口) 主实验评估 — 与经典基线 _baseline_hr.py 完全同协议.

覆盖:
  1. ubfc_only 模型 → UBFC test 40-49          (公平协议行)
  2. ubfc_pure 模型 → UBFC test 40-49          (联合训练行)
  3. ubfc_pure 模型 → PURE test 109-110
  4. ubfc_pure 模型 → MMPD 全部 51-83          (leave_mmpd 跨域, 统一口径)
  5. mmpd_dual_pt 模型 → MMPD test 78-83       (MMPD 主行)
用法: CUDA_VISIBLE_DEVICES=1 python3 -u _eval_step30.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
from evaluate import evaluate

config.TRAIN_CONFIG["VAL_STEP"] = 30   # 与基线一致的重叠窗口

UBFC_TEST = list(range(40, 50))
PURE_TEST = list(range(109, 111))
MMPD_TEST = list(range(78, 84))
MMPD_ALL = list(range(51, 84))

MODEL = {
    "ubfc_only": os.path.join(PROJ, "outputs", "ubfc_only", "remotessm_ubfc_only_best.pth"),
    "ubfc_pure": os.path.join(PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth"),
    "mmpd_dual": os.path.join(PROJ, "outputs", "mmpd_dual_pt", "remotessm_mmpd_dual_pt_best.pth"),
}


def run(name, model_key, subjects, roi, cache_v):
    config.TRAIN_CONFIG["FACE_ROI_SIZE"] = roi
    config.TRAIN_CONFIG["CACHE_VERSION"] = cache_v
    print(f"\n{'='*56}\n[{name}] model={model_key} roi={roi} cache={cache_v}\n{'='*56}")
    evaluate(model_path=MODEL[model_key], test_subjects=subjects)


def main():
    # 32x32 模型: UBFC/PURE (cardio_v5_bvphr)
    run("UBFC-only → UBFC test (step30)", "ubfc_only", UBFC_TEST, 32, "cardio_v5_bvphr")
    run("UBFC+PURE → UBFC test (step30)", "ubfc_pure", UBFC_TEST, 32, "cardio_v5_bvphr")
    run("UBFC+PURE → PURE test (step30)", "ubfc_pure", PURE_TEST, 32, "cardio_v5_bvphr")
    run("leave_mmpd: UBFC+PURE → MMPD ALL (step30)", "ubfc_pure", MMPD_ALL, 32, "cardio_v5_bvphr")
    # 64x64 模型: MMPD 双路 (cardio_v6_roi64)
    run("MMPD dual → MMPD test (step30)", "mmpd_dual", MMPD_TEST, 64, "cardio_v6_roi64")


if __name__ == "__main__":
    main()
