# pyrefly: ignore [missing-import]
import torch
print("MPS available:", torch.backends.mps.is_available())
