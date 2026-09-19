"""MMPD 复数→实数 SSM 消融 (运动域架构分析).

与 mmpd_facemain_pt (face-only 预训练微调, test 16.28) 完全同设置,
唯一差异: REAL_ONLY=1 冻结虚部 (A_imag/B_imag/C_imag=0).
背景: UBFC 静态场景输入信号强+残差兜底, 复数差异被掩盖;
      MMPD FACE_ONLY 下残差=0, 输出全靠 SSM 虚部振荡生成周期性波形,
      实数 SSM 将无法自激振荡, 差异会显著体现.

用法: CUDA_VISIBLE_DEVICES=0 python run_mmpd_faceonly_pt_real.py
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
    os.environ["FACE_ONLY"] = "1"
    os.environ["REAL_ONLY"] = "1"   # 消融: 纯实数 SSM
    os.environ["PRETRAIN_FULL"] = os.path.join(
        PROJ, "outputs", "ubfc_pure", "remotessm_ubfc_pure_best.pth")
    out_dir = os.path.join(PROJ, "outputs", "mmpd_real_pt")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MMPD-REAL-PT] train={len(TRAIN)} val={len(VAL)} test={len(TEST)} (REAL_ONLY=1)")
    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag="mmpd_real_pt")
    best_path = os.path.join(out_dir, "remotessm_mmpd_real_pt_best.pth")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
