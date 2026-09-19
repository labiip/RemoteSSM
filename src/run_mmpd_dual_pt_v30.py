"""MMPD 主行补跑: dual-path complex SSM, val_step=30 (统一选点口径).

背景: Table 4 四行原本选点标准不一致 ——
  主行 mmpd_dual_pt (13.55)          用 val_step=300 (495 窗) 选最佳 epoch
  消融 mmpd_dual_real_pt (13.74)     用 val_step=30  (4324 窗)
  消融 mmpd_dual_nores_pt (13.31)    用 val_step=30
  消融 mmpd_dual_nocob_pt (本次新增) 用 val_step=30
为消除该内部不一致, 本脚本以 mmpd_facemain_pt_dual.py (主行原始脚本) 为模板,
唯一差异: VAL_STEP = 30, 使四行的"最佳 epoch 选择标准"与测试口径完全一致.

与主行的差异仅此一处: 训练集(51-72)/验证集(73-77)/测试集(78-83)、
batch4/accum4/lr5e-5/40epoch/预训练微调/超参 全部相同.
训练数据 (train_step=30) 不受影响, 故训练过程本身与主行等价.

输出目录另开 mmpd_dual_pt_v30, 不覆盖原主行结果.

用法: CUDA_VISIBLE_DEVICES=0 python3 -u run_mmpd_dual_pt_v30.py
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
config.TRAIN_CONFIG["VAL_STEP"] = 30   # <-- 与三个消融行统一

from train import train_model
from evaluate import evaluate

TRAIN = list(range(51, 73))
VAL = list(range(73, 78))
TEST = list(range(78, 84))


def main():
    os.environ["PRETRAIN_FULL"] = os.path.join(
        PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth")
    out_dir = os.path.join(PROJ, "outputs", "mmpd_dual_pt_v30")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-DUAL-PT-V30] train={len(TRAIN)} val={len(VAL)} test={len(TEST)} (VAL_STEP=30)")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir,
                tag="mmpd_dual_pt_v30")
    best_path = os.path.join(out_dir, "remotessm_mmpd_dual_pt_v30_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
