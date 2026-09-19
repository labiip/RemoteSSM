"""UBFC 消融实验 runner (UBFC+PURE 联合训练, 与主实验同设置, 40 epoch).

用法:
  CUDA_VISIBLE_DEVICES=0 python run_ubfc_ablate.py no_residual
  CUDA_VISIBLE_DEVICES=0 python run_ubfc_ablate.py bpm_supervision
  CUDA_VISIBLE_DEVICES=0 python run_ubfc_ablate.py no_cob
  CUDA_VISIBLE_DEVICES=0 python run_ubfc_ablate.py no_outband

每个消融在 GPU0 上跑, 与 MMPD 主实验 (GPU1) 并行互不干扰.
"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

TRAIN = list(range(1, 35)) + list(range(101, 107))   # UBFC 1-34 + PURE 101-106
VAL = list(range(35, 40)) + list(range(107, 109))
TEST = list(range(40, 50))                           # UBFC test

ABLATIONS = {
    # name: (env_vars dict, config_overrides dict)
    "no_residual": ({"NO_RESIDUAL": "1"}, {}),
    "bpm_supervision": ({"BPM_TEMP": "0.5"}, {"WAVE_LOSS_W": 0.0, "BPM_LOSS_W": 3.0, "FREQ_LOSS_W": 0.0}),
    "no_cob": ({"RANDOM_A": "1"}, {}),
    "no_outband": ({}, {"OUTBAND_LOSS_W": 0.0}),
    "real_ssm": ({"REAL_ONLY": "1"}, {}),   # 复数→纯实数 SSM (冻结虚部 A_imag/B_imag/C_imag)
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ABLATIONS:
        print(f"usage: python run_ubfc_ablate.py {{{','.join(ABLATIONS)}}}")
        sys.exit(1)
    name = sys.argv[1]
    env_vars, cfg_overrides = ABLATIONS[name]

    for k, v in env_vars.items():
        os.environ[k] = v
    for k, v in cfg_overrides.items():
        config.TRAIN_CONFIG[k] = v

    config.TRAIN_CONFIG["EPOCHS"] = 40
    config.TRAIN_CONFIG["CURRICULUM_SEQ_LENS"] = [300]
    config.TRAIN_CONFIG["CURRICULUM_EPOCHS"] = [40]

    from train import train_model
    from evaluate import evaluate

    out_dir = os.path.join(PROJ, "outputs", f"ubfc_ablate_{name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[UBFC-ABLATE:{name}] train={len(TRAIN)} val={len(VAL)} test={len(TEST)} | env={env_vars} cfg={cfg_overrides}")

    train_model(train_subjects=TRAIN, val_subjects=VAL, out_dir=out_dir, tag=f"ubfc_ablate_{name}")
    best_path = os.path.join(out_dir, f"remotessm_ubfc_ablate_{name}_best.pth")
    print(f"\n[EVAL] {name} UBFC test 40-49")
    evaluate(model_path=best_path, test_subjects=TEST)


if __name__ == "__main__":
    main()
