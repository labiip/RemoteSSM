import torch
print("torch", torch.__version__)
print("cap", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
a = torch.randn(1000, 1000, device="cuda")
b = (a @ a).sum().item()
print("matmul-ok", round(b, 2))
import mamba_ssm  # noqa: E402
print("mamba-import-ok", mamba_ssm.__version__)
