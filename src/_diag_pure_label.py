"""PURE test 标签口径诊断: dataset 的 target_hr vs BVP numpy FFT HR.

若差异大 → 标签口径问题 (extractor/HR范围/对齐), 模型 22.45 可能是伪差.
若一致 → 模型输出问题, 需查模型在 PURE 上的行为.
用法: python3 _diag_pure_label.py
"""
import os, sys
import numpy as np
import torch

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from config import COMMON_CONFIG, TRAIN_CONFIG, PHYSIO_CONFIG
from dataset import TitanrPPGDataset
from utils import DifferentiablePhysioExtractor

FS = COMMON_CONFIG["FS"]
LO, HI = PHYSIO_CONFIG["HR_BAND"]


def fft_hr_np(sig):
    sig = np.asarray(sig, np.float32)
    if len(sig) < 16:
        return np.nan
    sig = sig - sig.mean()
    s = sig / (sig.std() + 1e-8)
    sw = s * np.hanning(len(sig))
    nfft = max(1024, 1 << int(np.ceil(np.log2(len(sig)))))
    spec = np.abs(np.fft.rfft(sw, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / FS)
    m = (freqs >= LO) & (freqs <= HI)
    if not m.any():
        return np.nan
    return freqs[m][np.argmax(spec[m])] * 60.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TitanrPPGDataset(root_dir=TRAIN_CONFIG["DATASET_ROOT"], seq_len=300, step=300,
                          subject_ids=[109, 110], split_name="test")
    print(f"PURE test 样本数: {len(ds)}")
    extractor = DifferentiablePhysioExtractor(fs=FS).to(device)

    tgt, np_fft, ext = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        bvp = s["target_bvp"]                     # (300,)
        thr = s["target_hr"]
        t_fft = fft_hr_np(bvp)
        tb = torch.tensor(bvp, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            hr_ext = extractor(tb, freq_range=(LO, HI), temperature=0.05).cpu().item()
        tgt.append(thr)
        np_fft.append(t_fft)
        ext.append(hr_ext)

    tgt = np.array(tgt, np.float32)
    np_fft = np.array(np_fft, np.float32)
    ext = np.array(ext, np.float32)
    m = np.isfinite(np_fft)
    print(f"target_hr:    mean={tgt.mean():.1f} std={tgt.std():.1f}")
    print(f"numpy FFT HR: mean={np_fft[m].mean():.1f} std={np_fft[m].std():.1f}")
    print(f"extractor HR: mean={ext.mean():.1f} std={ext.std():.1f}")
    print(f"target vs numpyFFT: MAE={np.mean(np.abs(tgt[m]-np_fft[m])):.2f} r={np.corrcoef(tgt[m], np_fft[m])[0,1]:.3f}")
    print(f"target vs extractor: MAE={np.mean(np.abs(tgt-ext)):.2f} r={np.corrcoef(tgt, ext)[0,1]:.3f}")
    print(f"numpyFFT vs extractor: MAE={np.mean(np.abs(np_fft[m]-ext[m])):.2f}")


if __name__ == "__main__":
    main()
