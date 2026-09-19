"""MMPD face_roi-only 迁移微调: 从 UBFC+PURE 预训练权重初始化.

背景: 随机初始化训 MMPD 发生 HR 输出坍缩 (输出恒定 ~73 BPM, test r=0.04).
原因: 模型从未见过干净脉搏样本, 运动噪声下坍缩到训练集均值先验.
方案: 加载 UBFC+PURE (r=0.995) 完整权重 (含 FrameStem) 作初始化,
      带着真实脉搏知识适配 MMPD 运动域. FrameStem 卷积核与分辨率无关,
      32x32 权重可直接放入 64x64 模型 (shape 完全兼容).

用法:
  PRETRAIN_FULL=... CUDA_VISIBLE_DEVICES=1 python run_mmpd_faceonly_pt.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

# 64x64 face_roi + 复用已有 cache (cardio_v6_roi64 已构建, 不重建)
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
config.TRAIN_CONFIG["BATCH_SIZE"] = 4
config.TRAIN_CONFIG["GRAD_ACCUM"] = 4
config.TRAIN_CONFIG["LEARNING_RATE"] = 5e-5   # 微调减半 LR, 避免破坏预训练知识

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
    os.environ["PRETRAIN_FULL"] = os.path.join(
        PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth")
    out_dir = os.path.join(PROJ, "outputs", "mmpd_facemain_pt")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-FACEMAIN-PT] train={len(TRAIN)} val={len(VAL)} test={len(TEST)}")
    print(f"[MMPD-FACEMAIN-PT] PRETRAIN_FULL={os.environ['PRETRAIN_FULL']}")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="mmpd_facemain_pt")
    best_path = os.path.join(out_dir, "remotessm_mmpd_facemain_pt_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
