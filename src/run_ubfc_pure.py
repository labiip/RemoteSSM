"""UBFC+PURE 联合训练主实验 runner (subject-independent).

解决 UBFC 数据量小的问题: 加入 PURE (6 训练 subject) 扩充训练集.
全局 subject id (与 run_lodo.py 一致):
  UBFC: 1-49,  PURE: 101-110
划分:
  train = UBFC 1-34 + PURE 101-106 (40 subject)
  val   = UBFC 35-39 + PURE 107-108
  test  = UBFC 40-49 (单独评估) + PURE 109-110 (单独评估)

用法:
  CUDA_VISIBLE_DEVICES=3 python run_ubfc_pure.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# step=10 (复用已有缓存, 样本更多: UBFC 3338 + PURE ~数千)
config.TRAIN_CONFIG["EPOCHS"] = 40
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [40]

from train import train_model
from evaluate import evaluate

TRAIN = list(range(1, 35)) + list(range(101, 107))   # UBFC 1-34 + PURE 101-106
VAL = list(range(35, 40)) + list(range(107, 109))    # UBFC 35-39 + PURE 107-108
TEST_UBFC = list(range(40, 50))                      # UBFC 40-49
TEST_PURE = list(range(109, 111))                    # PURE 109-110


def main():
    out_dir = os.path.join(PROJ, "outputs", "ubfc_pure")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[UBFC+PURE] train={len(TRAIN)} val={len(VAL)}")
    print(f"[UBFC+PURE] test: UBFC={len(TEST_UBFC)} PURE={len(TEST_PURE)}")

    train_model(train_subjects=TRAIN, val_subjects=VAL,
                out_dir=out_dir, tag="ubfc_pure")

    best_path = os.path.join(out_dir, "remotessm_ubfc_pure_best.pth")
    print("\n[EVAL] UBFC test 40-49")
    evaluate(model_path=best_path, test_subjects=TEST_UBFC)
    print("\n[EVAL] PURE test 109-110")
    evaluate(model_path=best_path, test_subjects=TEST_PURE)


if __name__ == "__main__":
    main()
