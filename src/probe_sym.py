#!/usr/bin/env python
# 探测 c10/c10_cuda/at 相关缺失符号 (忽略由 python 解释器提供的 Py* 符号)
import os, sys, glob
try:
    from elftools.elf.elffile import ELFFile
except ImportError:
    os.system(sys.executable + " -m pip install pyelftools -q")
    from elftools.elf.elffile import ELFFile

VENV = os.path.expanduser("~/mamba_env")
TLIB = os.path.join(VENV, "lib/python3.12/site-packages/torch/lib")

def syms(path, undef_only):
    out = {}
    try:
        with open(path, "rb") as f:
            elf = ELFFile(f)
            ds = elf.get_section_by_name(".dynsym")
            if not ds:
                return out
            for sym in ds.iter_symbols():
                if not sym.name:
                    continue
                if undef_only:
                    if sym["st_shndx"] == "SHN_UNDEF":
                        out[sym.name] = True
                else:
                    if sym["st_shndx"] != "SHN_UNDEF":
                        out[sym.name] = True
    except Exception as e:
        print("skip", path, e)
    return out

so = sys.argv[1]
need = syms(so, undef_only=True)

have = {}
# torch libs (libc10*, libtorch*) 排除 _cpu/_cuda 时仍保留
for lp in sorted(glob.glob(os.path.join(TLIB, "libc10*.so")) +
                 glob.glob(os.path.join(TLIB, "libtorch*.so"))):
    have.update(syms(lp, undef_only=False))

# 关注 c10/c10_cuda/at/gil 等核心库符号(排除 Py* python API 与 libstdc++/cuda driver 符号)
skip_prefix = ("Py", "std::", "__", "_ZSt", "nv", "cuda", "cublas", "curand", "cudnn", "cu", "dlopen", "dlsym")
def interesting(n):
    if n.startswith(skip_prefix):
        return False
    return ("c10" in n) or n.startswith("_ZN3c10") or n.startswith("_ZN2at") or n.startswith("_ZN3at") or n.startswith("_ZN3c10") or ("Warning" in n) or ("cuda_check" in n) or n.startswith("at::") or n.startswith("c10::")

missing = sorted(n for n in need if (n not in have) and interesting(n))
print("TARGET:", os.path.basename(so))
print("INTERESTING_MISSING:", len(missing))
for m in missing[:40]:
    print("  MISS:", m)
if not missing:
    print("NO-C10-MISSING")
