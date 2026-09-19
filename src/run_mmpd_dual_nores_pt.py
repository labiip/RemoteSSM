"""MMPD 双路消融 A: dual-path complex SSM 去掉残差连接 (NO_RESIDUAL=1).

与 mmpd_facemain_pt_dual (dual-path complex, test 13.55) 完全同设置
(64x64 + UBFC+PURE 预训练微调 + batch4/accum4 + 40 epoch),
唯一差异: NO_RESIDUAL=1 输出不再叠加 CHROM 输入 (bvp_pred = SSM 精修信号).

目的: 在运动域隔离"残差连接"的贡献 (静态 UBFC 上残差基本无影响,
      残差作为运动下保底机制的贡献需在 dual-path 设置下单独验证).
评估: 与主行一致用 step=30 重叠窗口 (VAL_STEP=30), test 78-83.

用法: CUDA_VISIBLE_DEVICES=1 python run_mmpd_dual_nores_pt.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
config.TRAIN_CONFIG["BATCH_SIZE"] = 4
config.TRAIN_CONFIG["GRAD_ACCUM"] = 4
config.TRAIN_CONFIG["LEARNING_RATE"] = 5e-5

config.TRAIN_CONFIG["TRAIN_STEP"] = 30
config.TRAIN_CONFIG["EPOCHS"] = 40
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [40]
config.TRAIN_CONFIG["VAL_STEP"] = 30   # 与 _eval_step30.py 主行口径一致

from train import train_model
from evaluate import evaluate

TRAIN = list(range(51, 73))
VAL = list(range(73, 78))
TEST = list(range(78, 84))


def main():
    os.environ["NO_RESIDUAL"] = "1"   # 消融: 去掉残差连接 (双路保留)
    os.environ["PRETRAIN_FULL"] = os.path.join(
        PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth")
    out_dir = os.path.join(PROJ, "outputs", "mmpd_dual_nores_pt")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-DUAL-NORES-PT] train={len(TRAIN)} val={len(VAL)} test={len(TEST)} (NO_RESIDUAL=1)")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="mmpd_dual_nores_pt")
    best_path = os.path.join(out_dir, "remotessm_mmpd_dual_nores_pt_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
