"""三个 checkpoint 的 step-30 统一口径评测 (与论文表格同协议).

含"训练已完成"守卫: 只有 .out 里出现 "Done. Best Val FFT"(train_model 末尾才打印)
才评测, 避免误测中途保存的 best.pth (best.pth 每次 val 改善都会覆盖写).

1. UBFC 复数 vs 实数 SSM   -> outputs/ubfc_ablate_real_ssm
   roi=32, cache=cardio_v5_bvphr, test 40-49, VAL_STEP=30 (481 窗)
2. MMPD 有 COB vs 无 COB   -> outputs/mmpd_dual_nocob_pt
   roi=64, cache=cardio_v6_roi64, test 78-83, VAL_STEP=30
3. MMPD 主行 (val_step=30) -> outputs/mmpd_dual_pt_v30
   同上口径, 用于统一四行的选点标准

用法: CUDA_VISIBLE_DEVICES=3 python3 -u eval_new_ablations_step30.py
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config
from evaluate import evaluate

OUT = os.path.join(PROJ, "outputs")
UBFC_TEST = list(range(40, 50))
MMPD_TEST = list(range(78, 84))


def training_done(tag):
    """训练是否真正跑完: train_model 末尾会打印 'Done. Best Val FFT'."""
    for cand in (os.path.join(PROJ, f"{tag}.out"),
                 os.path.join(PROJ, f"{tag}_train.log")):
        if os.path.exists(cand):
            with open(cand, "r", errors="ignore") as f:
                if "Done. Best Val FFT" in f.read():
                    return True
    # 回退: 训练 log 里出现 40/40 也算完成
    lg = os.path.join(OUT, tag, f"remotessm_{tag}_train.log")
    if os.path.exists(lg):
        with open(lg, "r", errors="ignore") as f:
            txt = f.read()
            if "Epoch  40/40" in txt or "Epoch 40/40" in txt:
                return True
    return False


JOBS = [
    # (标题, tag, ckpt文件名, roi, cache, test_subjects, out文件名)
    ("UBFC real-valued SSM -> UBFC test (step30)",
     "ubfc_ablate_real_ssm", "remotessm_ubfc_ablate_real_ssm_best.pth",
     32, "cardio_v5_bvphr", UBFC_TEST, "ubfc_ablate_real_ssm.out"),

    ("MMPD dual w/o COB -> MMPD test (step30)",
     "mmpd_dual_nocob_pt", "remotessm_mmpd_dual_nocob_pt_best.pth",
     64, "cardio_v6_roi64", MMPD_TEST, "mmpd_dual_nocob_pt.out"),

    ("MMPD dual main (val_step30) -> MMPD test (step30)",
     "mmpd_dual_pt_v30", "remotessm_mmpd_dual_pt_v30_best.pth",
     64, "cardio_v6_roi64", MMPD_TEST, "mmpd_dual_pt_v30.out"),
]

for title, tag, ckpt, roi, cache, test_subj, outname in JOBS:
    p = os.path.join(OUT, tag, ckpt)
    if not os.path.exists(p):
        print(f"\n[SKIP] {tag}: checkpoint 不存在 ({p})", flush=True)
        continue
    if not training_done(tag):
        print(f"\n[SKIP] {tag}: 训练尚未完成 (未见 'Done. Best Val FFT'), "
              f"避免误测中途 best.pth", flush=True)
        continue

    config.TRAIN_CONFIG["FACE_ROI_SIZE"] = roi
    config.TRAIN_CONFIG["CACHE_VERSION"] = cache
    config.TRAIN_CONFIG["VAL_STEP"] = 30

    print("\n" + "=" * 60)
    print(f"########## {title} ##########", flush=True)
    print("=" * 60)
    evaluate(model_path=p, test_subjects=test_subj)

print("\nALL AVAILABLE EVAL DONE", flush=True)
