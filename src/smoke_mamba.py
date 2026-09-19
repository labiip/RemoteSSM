"""Mamba 模型 smoke test (服务器 venv): 前向形状/参数量/反向/NaN 检查"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cudnn.enabled = False  # cuDNN9 无 Pascal(sm61) 3D-conv engine, 走原生实现

# PhysMamba 需要 Bi-Mamba -> 替换 mamba_ssm.Mamba 顶层导出
import mamba_ssm
from vendor_bimamba import MambaBi
mamba_ssm.Mamba = MambaBi

from physmamba_model import PhysMamba
from rhythmmamba_model import RhythmMamba

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# ---------- PhysMamba ----------
pm = PhysMamba(frames=300).to(dev)
np_pm = sum(p.numel() for p in pm.parameters()) / 1e6
print(f"PhysMamba params: {np_pm:.2f}M")
x1 = torch.randn(2, 3, 300, 32, 32, device=dev)  # [B,C,T,H,W]
y1 = pm(x1)
print("PhysMamba out:", tuple(y1.shape))
assert y1.shape == (2, 300), "PhysMamba output length != 300"
loss = (y1 - torch.randn_like(y1)).pow(2).mean()
loss.backward()
assert torch.isfinite(loss), "PhysMamba loss NaN"
print("PhysMamba fwd+bwd OK")

# ---------- RhythmMamba ----------
rm = RhythmMamba(depth=24, embed_dim=96).to(dev)
np_rm = sum(p.numel() for p in rm.parameters()) / 1e6
print(f"RhythmMamba params: {np_rm:.2f}M")
x2 = torch.randn(1, 300, 3, 32, 32, device=dev)  # [N,D,C,H,W]
y2 = rm(x2)
print("RhythmMamba out:", tuple(y2.shape))
assert y2.shape == (1, 300), "RhythmMamba output length != 300"
loss2 = (y2 - torch.randn_like(y2)).pow(2).mean()
loss2.backward()
assert torch.isfinite(loss2), "RhythmMamba loss NaN"
print("RhythmMamba fwd+bwd OK")
print("SMOKE-ALL-OK")
