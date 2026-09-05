"""Device selection helpers for training and generation."""

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to CUDA, then MPS, then CPU.

    CUDA is preferred when available. Apple Silicon MPS is used when PyTorch was
    built with MPS and the backend is available. Otherwise training runs on CPU.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
