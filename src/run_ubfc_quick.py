"""UBFC-only 快速验证 runner: 验证残差连接是否让 MAE 逼近 CHROM/POS 下界 (~3.2 BPM).

用法:
  CUDA_VISIBLE_DEVICES=3 python run_ubfc_quick.py

独立 out_dir + tag, 不与主实验 checkpoint 冲突。限 20 epoch, 单阶段 curriculum。
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# 快速验证: 限 epoch + 单阶段 curriculum (GRL warmup=30 > 20, 自动不激活对抗)
config.TRAIN_CONFIG["EPOCHS"] = 20
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [20]

from train import train_model
from evaluate import evaluate

TRAIN = list(range(1, 35))   # UBFC 1-34
VAL = list(range(35, 40))    # UBFC 35-39
TEST = list(range(40, 50))   # UBFC 40-49


def main():
    out_dir = os.path.join(PROJ, "outputs", "ubfc_quick")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[UBFC-QUICK] train={len(TRAIN)} val={len(VAL)} test={len(TEST)}")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="ubfc_quick")
    best_path = os.path.join(out_dir, "remotessm_ubfc_quick_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()