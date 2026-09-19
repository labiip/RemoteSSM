"""PURE-only intra-dataset 主实验 runner (subject-independent).

PURE = subjects 101-110 (10 subject, 每 subject 多段视频).
划分 (与 run_lodo.py 全局 subject id 一致):
  train 101-106, val 107-108, test 109-110

用法:
  CUDA_VISIBLE_DEVICES=0 python run_pure_own.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# 快速验证配置: 与 MMPD 对齐, 先 20 epoch 看方向
config.TRAIN_CONFIG["EPOCHS"] = 20
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [20]

from train import train_model
from evaluate import evaluate

TRAIN = list(range(101, 107))   # PURE 101-106
VAL = list(range(107, 109))     # PURE 107-108
TEST = list(range(109, 111))    # PURE 109-110


def main():
    out_dir = os.path.join(PROJ, "outputs", "pure_own")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[PURE-OWN] train={len(TRAIN)} val={len(VAL)} test={len(TEST)}")
    print(f"[PURE-OWN] visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    train_model(train_subjects=TRAIN, val_subjects=VAL,
                out_dir=out_dir, tag="pure_own")

    best_path = os.path.join(out_dir, "remotessm_pure_own_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()