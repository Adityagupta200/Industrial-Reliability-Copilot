import os
import builtins

# 1. Force HuggingFace to acknowledge PyTorch
os.environ["USE_TORCH"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import torch.nn as nn
import torch.optim.lr_scheduler

# 2. Comprehensively patch missing tensor types for older PyTorch builds
for missing_type, fallback_type in [
    ("uint16", "int16"),
    ("uint32", "int32"),
    ("uint64", "int64"),
    ("float8_e4m3fn", "float32"),
    ("float8_e5m2", "float32"),
    ("float8_e4m3fnuz", "float32"),
    ("float8_e5m2fnuz", "float32"),
    ("bfloat16", "float16"),
]:
    if not hasattr(torch, missing_type):
        setattr(torch, missing_type, getattr(torch, fallback_type, torch.float32))

# 3. Polyfill LRScheduler for PyTorch < 2.0 compatibility
if hasattr(torch.optim.lr_scheduler, "_LRScheduler"):
    LRScheduler = torch.optim.lr_scheduler._LRScheduler
else:
    LRScheduler = object

setattr(torch.optim.lr_scheduler, "LRScheduler", LRScheduler)

# 4. Bind to builtins to ensure all downstream modules use the patched versions
builtins.torch = torch
builtins.nn = nn
builtins.LRScheduler = LRScheduler
