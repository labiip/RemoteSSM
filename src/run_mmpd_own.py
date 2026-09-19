"""MMPD-only intra-dataset 主实验 runner (subject-independent).

用法:
  CUDA_VISIBLE_DEVICES=1 python run_mmpd_own.py

独立 out_dir + tag, 不与 UBFC-only (GPU0, 默认 remotessm_*) 的 checkpoint 冲突。
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# MMPD 正式主实验: 双路 (face_roi + BGR), GRL 已关 (config), 30 epoch 看 val 曲线定论
config.TRAIN_CONFIG["TRAIN_STEP"] = 30
config.TRAIN_CONFIG["EPOCHS"] = 30
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [30]

from train import train_model
from evaluate import evaluate

TRAIN = list(range(51, 73))   # MMPD 51-72
VAL = list(range(73, 78))     # MMPD 73-77
TEST = list(range(78, 84))    # MMPD 78-83


def main():
    out_dir = os.path.join(PROJ, "outputs", "mmpd_own")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-OWN] train={len(TRAIN)} val={len(VAL)} test={len(TEST)}")
    print(f"[MMPD-OWN] visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    train_model(train_subjects=TRAIN, val_subjects=VAL,
                out_dir=out_dir, tag="mmpd_own")

    best_path = os.path.join(out_dir, "remotessm_mmpd_own_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()