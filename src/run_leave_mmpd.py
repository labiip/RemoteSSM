"""leave_mmpd: LODO 跨域补充实验.

用已训好的 UBFC+PURE 模型 (32x32, cardio_v5_bvphr) 评估 MMPD 全部 33 个受试者
(51-83), 零训练成本. 论文中作为泛化性补充分析 (跨域多变量同时变化, 不作主证据).

用法: CUDA_VISIBLE_DEVICES=0 python run_leave_mmpd.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# UBFC+PURE 模型是 32x32, 保持与训练一致
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 32
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v5_bvphr"
config.TRAIN_CONFIG["VAL_STEP"] = 300

from evaluate import evaluate

MMPD_ALL = list(range(51, 84))   # 整块留出: 全部 33 个 MMPD 受试者


def main():
    best_path = os.path.join(PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth")
    print(f"[LODO-leave_mmpd] 用 UBFC+PURE 模型评估 MMPD 全部 {len(MMPD_ALL)} 受试者")
    evaluate(model_path=best_path, test_subjects=MMPD_ALL)


if __name__ == "__main__":
    main()
