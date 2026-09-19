import os, subprocess, glob, shutil
print("python", __import__("sys").version)
try:
    import torch
    print("torch", torch.__version__, "cuda-build", torch.version.cuda,
          "avail", torch.cuda.is_available(),
          "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
except Exception as e:
    print("torch import fail:", e)
for pkg in ("mamba_ssm", "causal_conv1d"):
    try:
        m = __import__(pkg)
        print(pkg, "OK", getattr(m, "__version__", "?"))
    except Exception as e:
        print(pkg, "NOT-INSTALLED", type(e).__name__)
print("nvcc candidates:", glob.glob("/usr/local/cuda*/bin/nvcc"))
print("conda candidates:", glob.glob(os.path.expanduser("~/.conda/bin/conda"))
      + glob.glob(os.path.expanduser("~/miniconda3/bin/conda"))
      + glob.glob(os.path.expanduser("~/anaconda3/bin/conda")))
try:
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
    for ln in out.stdout.splitlines():
        if "Driver" in ln or "CUDA Version" in ln:
            print(ln)
except Exception as e:
    print("nvidia-smi fail", e)
