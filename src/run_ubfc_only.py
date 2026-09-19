"""UBFC-only 训练 (公平对比协议).

与 UBFC+PURE 联合训练同配置, 仅训练集只用 UBFC 1-34 (27 subject, 标准文献协议).
用途: 主对比表与文献数字同协议对比; 联合训练作为"小样本训练策略"单独呈现.

用法: CUDA_VISIBLE_DEVICES=0 python run_ubfc_only.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

config.TRAIN_CONFIG["EPOCHS"] = 40
config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [40]

from train import train_model
from evaluate import evaluate

TRAIN = list(range(1, 35))    # UBFC 1-34 (27 train subjects, 标准协议)
VAL = list(range(35, 40))     # UBFC 35-39
TEST = list(range(40, 50))    # UBFC 40-49


def main():
    out_dir = os.path.join(PROJ, "outputs", "ubfc_only")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[UBFC-ONLY] train={len(TRAIN)} val={len(VAL)} test={len(TEST)} (单数据集公平协议)")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="ubfc_only")
    best_path = os.path.join(out_dir, "remotessm_ubfc_only_best.pth")
    print("\n[EVAL] UBFC test 40-49")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
