import os

import torch

# If PYGIP_ONLY_GPU is set (1/true/yes), require CUDA; otherwise fall back to cpu when CUDA is not available.
_only_gpu = os.getenv("PYGIP_ONLY_GPU", "").lower() in ("1", "true", "yes")
if _only_gpu and not torch.cuda.is_available():
    raise RuntimeError(
        "PYGIP_ONLY_GPU is set but no CUDA-capable device is available. "
        "Unset PYGIP_ONLY_GPU or enable a CUDA device/driver to proceed."
    )

# Allow explicit override via PYGIP_DEVICE; otherwise prefer CUDA if available.
_DEFAULT_DEVICE_STR = os.getenv("PYGIP_DEVICE")
if _DEFAULT_DEVICE_STR:
    _default_device = torch.device(_DEFAULT_DEVICE_STR)
else:
    if _only_gpu:
        # _only_gpu implies CUDA is available (checked above)
        _DEFAULT_DEVICE_STR = "cuda:0"
    else:
        _DEFAULT_DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu"
    _default_device = torch.device(_DEFAULT_DEVICE_STR)


def get_device():
    return _default_device


def set_device(device_str):
    global _default_device
    _default_device = torch.device(device_str)
