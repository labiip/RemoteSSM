"""UBFC+PURE 波形监督快速验证 (10 epoch).

目的: 验证去掉 BPM 数值监督、改用波形(WAVE)+频谱(FREQ)监督后,
模型输出 BVP 的 FFT 峰值 HR 是否对齐 GT (预测均值≈GT 均值, MAE 显著下降).

用法:
  CUDA_VISIBLE_DEVICES=0 python run_ubfc_pure_quick.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

config.TRAIN_CONFIG["EPOCHS"] = 10
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [10]

from train import train_model
from evaluate import evaluate

TRAIN = list(range(1, 35)) + list(range(101, 107))
VAL = list(range(35, 40)) + list(range(107, 109))
TEST_UBFC = list(range(40, 50))
TEST_PURE = list(range(109, 111))


def main():
    out_dir = os.path.join(PROJ, "outputs", "ubfc_pure_quick")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[UBFC+PURE-WAVE-QUICK] train={len(TRAIN)} val={len(VAL)}")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="ubfc_pure_quick")
    best_path = os.path.join(out_dir, "remotessm_ubfc_pure_quick_best.pth")
    print("\n[EVAL] UBFC test 40-49")
    evaluate(model_path=best_path, test_subjects=TEST_UBFC)
    print("\n[EVAL] PURE test 109-110")
    evaluate(model_path=best_path, test_subjects=TEST_PURE)


if __name__ == "__main__":
    main()
