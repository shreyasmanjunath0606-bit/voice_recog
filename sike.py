"""Phase 0: Cross-platform hardware smoke test.

Automatically detects Apple Silicon MPS, NVIDIA CUDA, or CPU fallback.
"""

import platform
import torch

print("========================================")
print(" Hardware & PyTorch Smoke Test (Phase 0)")
print("========================================")
print(f"OS / System     : {platform.system()} ({platform.machine()})")
print(f"PyTorch Version : {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Compute Backend : ✅ NVIDIA CUDA ({gpu_name})")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Compute Backend : ✅ Apple Silicon Metal (MPS)")
else:
    device = torch.device("cpu")
    print("Compute Backend : ℹ️ CPU Fallback (Universal)")

# Test tensor creation and computation on selected device
x = torch.ones(3, 3, device=device)
y = x * 2
print(f"Tensor Test     : ✅ Executed successfully on {device}")
print("========================================")
