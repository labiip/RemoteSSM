"""MMPD 双路消融 C: dual-path complex SSM 去掉 COB 频率先验 (RANDOM_A=1).

与 mmpd_dual_pt (dual-path complex, test 13.55) 完全同设置
(64x64 + UBFC+PURE 预训练微调 + batch4/accum4 + 40 epoch),
唯一差异: RANDOM_A=1, 预训练权重加载后把 A 矩阵重新随机初始化,
         即去掉 COB 的心脏频段频率先验 (0.7-3.0 Hz), 其余权重与主行一致.

目的: 在运动域隔离 COB 频率先验的贡献. 此前该结论只在 UBFC 上测过
      (step30: full 0.87 -> w/o COB 0.59), MMPD 上尚未单独验证.
注意: A 不是冻结参数 (可学习), 故本消融衡量的是"初始化先验"的价值,
      而非"是否学习频率".

评估: 与主行一致用 step=30 重叠窗口 (VAL_STEP=30), test 78-83.

用法: CUDA_VISIBLE_DEVICES=1 python run_mmpd_dual_nocob_pt.py
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
    os.environ["RANDOM_A"] = "1"   # 消融: 去掉 COB 频率先验 (双路保留)
    os.environ["PRETRAIN_FULL"] = os.path.join(
        PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth")
    out_dir = os.path.join(PROJ, "outputs", "mmpd_dual_nocob_pt")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-DUAL-NOCOB-PT] train={len(TRAIN)} val={len(VAL)} test={len(TEST)} (RANDOM_A=1)")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir,
                tag="mmpd_dual_nocob_pt")
    best_path = os.path.join(out_dir, "remotessm_mmpd_dual_nocob_pt_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
