"""MMPD face_roi-only 快速验证: 信号支路置零, 验证模型能否从空间信息学心率.

诊断目的: 若 val r(HR) 能转正 (如 >0.2), 说明 face_roi 空间信息可学,
全量训练值得跑; 若仍≈0/负, 说明模型无法利用空间信息, 需换路线.

用法:
  FACE_ONLY=1 CUDA_VISIBLE_DEVICES=1 python run_mmpd_faceonly.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# 快速验证: step=30 (同 mmpd_own), 10 epoch
config.TRAIN_CONFIG["TRAIN_STEP"] = 30
config.TRAIN_CONFIG["EPOCHS"] = 10
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [10]

from train import train_model
from evaluate import evaluate

TRAIN = list(range(51, 73))
VAL = list(range(73, 78))
TEST = list(range(78, 84))


def main():
    os.environ["FACE_ONLY"] = "1"
    out_dir = os.path.join(PROJ, "outputs", "mmpd_faceonly")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-FACEONLY] train={len(TRAIN)} val={len(VAL)} test={len(TEST)}")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="mmpd_faceonly")
    best_path = os.path.join(out_dir, "remotessm_mmpd_faceonly_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
