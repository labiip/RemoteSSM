"""MMPD 双路 (face_roi + BGR) 对比消融.

与 mmpd_facemain_pt 完全同设置 (预训练微调 + 64x64 + batch4/accum4),
唯一差异: 不置零 BGR 信号支路 (FACE_ONLY 关闭), 验证 face-only vs 双路.
背景: 32x32 时代 BGR 支路是负资产 (val r=-0.11 vs face-only +0.18);
      64x64 + 预训练微调下重新验证.

用法: CUDA_VISIBLE_DEVICES=1 python run_mmpd_faceonly_pt_dual.py
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

from train import train_model
from evaluate import evaluate

TRAIN = list(range(51, 73))
VAL = list(range(73, 78))
TEST = list(range(78, 84))


def main():
    os.environ["PRETRAIN_FULL"] = os.path.join(
        PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth")
    out_dir = os.path.join(PROJ, "outputs", "mmpd_dual_pt")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-DUAL-PT] train={len(TRAIN)} val={len(VAL)} test={len(TEST)} (BGR 支路打开)")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="mmpd_dual_pt")
    best_path = os.path.join(out_dir, "remotessm_mmpd_dual_pt_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
