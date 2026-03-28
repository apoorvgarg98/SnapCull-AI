from __future__ import annotations

import os

import torch


def resolve_torch_device() -> str:
    """
    Resolve execution device for torch models.

    Supports optional GPU selection via `GPU_DEVICE_INDEX` (default: 0).
    Returns `cpu` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return "cpu"

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return "cpu"

    requested = os.getenv("GPU_DEVICE_INDEX", "0")
    try:
        device_index = int(requested)
    except ValueError:
        device_index = 0

    if device_index < 0 or device_index >= device_count:
        device_index = 0

    return f"cuda:{device_index}"


def cuda_diagnostics() -> str:
    """Return human-readable CUDA diagnostics for troubleshooting."""
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    names = [torch.cuda.get_device_name(i) for i in range(count)] if available else []
    return (
        f"torch={torch.__version__}, "
        f"torch_cuda_build={torch.version.cuda}, "
        f"cuda_available={available}, "
        f"cuda_device_count={count}, "
        f"cuda_devices={names}"
    )
