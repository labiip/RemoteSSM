# -*- coding: utf-8 -*-
"""?????checkpoint ??step30 ?????????, ????????? (???/real/???19.72)"""
import os, sys

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import config

config.TRAIN_CONFIG["VAL_STEP"] = 30

from evaluate import evaluate

OUT = os.path.join(PROJ, "outputs")
UBFC_TEST = list(range(40, 50))
MMPD_TEST = list(range(78, 84))
MMPD_ALL = list(range(51, 84))

# 1-3. UBFC ??? (roi=32, cardio_v5_bvphr ???)
for tag in ["no_residual", "bpm_supervision", "no_outband"]:
    p = os.path.join(OUT, f"ubfc_ablate_{tag}", f"remotessm_ubfc_ablate_{tag}_best.pth")
    print(f"\n########## ABLATE {tag} -> UBFC test (step30) ##########", flush=True)
    evaluate(model_path=p, test_subjects=UBFC_TEST)

# 4. ubfc_only ??MMPD ALL (???, roi=32 + cardio_v5_bvphr, ?????? 19.72)
p = os.path.join(OUT, "ubfc_only", "remotessm_ubfc_only_best.pth")
print(f"\n########## UBFC-only -> MMPD ALL (step30) ##########", flush=True)
evaluate(model_path=p, test_subjects=MMPD_ALL)

# 5. MMPD real-valued (roi=64, cardio_v6_roi64, ?????? 16.64)
config.TRAIN_CONFIG["FACE_ROI_SIZE"] = 64
config.TRAIN_CONFIG["CACHE_VERSION"] = "cardio_v6_roi64"
p = os.path.join(OUT, "mmpd_real_pt", "remotessm_mmpd_real_pt_best.pth")
print(f"\n########## MMPD real-ssm -> MMPD test (step30, roi64) ##########", flush=True)
evaluate(model_path=p, test_subjects=MMPD_TEST)

print("\nALL VERIFY DONE", flush=True)

