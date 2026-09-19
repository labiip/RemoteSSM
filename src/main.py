"""MMPD face_roi-only 正式主实验 (验证已证明有效: test MAE 13.85 vs 经典 28.46).

MMPD 上 BGR 支路 (CHROM/POS, r≈0.04) 是无效信号且拖累双路, 故正式主实验
采用 face_roi-only 模式 (FACE_ONLY=1, 信号支路置零), 模型纯靠空间支路学习.

用法:
  FACE_ONLY=1 CUDA_VISIBLE_DEVICES=0 python main.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# 64x64 face_roi (保留更多空间细节, 已验证信息量提升) + 独立 cache 版本避免与 32 冲突
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
config.TRAIN_CONFIG["BATCH_SIZE"] = 4   # 64x64 输入在 TITAN Xp 12GB 上 batch 8 会 OOM (实测), 降到 4
config.TRAIN_CONFIG["GRAD_ACCUM"] = 4   # 梯度累积 4 步: 等效 batch 4x4=16, 不增显存

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
    os.environ["FACE_ONLY"] = "1"
    out_dir = os.path.join(PROJ, "outputs", "mmpd_facemain")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-FACEMAIN] train={len(TRAIN)} val={len(VAL)} test={len(TEST)}")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="mmpd_facemain")
    best_path = os.path.join(out_dir, "remotessm_mmpd_facemain_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
