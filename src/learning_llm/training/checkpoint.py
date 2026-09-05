"""Checkpoint save/load helpers."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from learning_llm.model import DecoderLanguageModel, ModelConfig


def save_checkpoint(
    path: str | Path,
    *,
    model: DecoderLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save enough state to resume training without resetting the optimizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(model.config),
            "step": step,
            "epoch": epoch,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[DecoderLanguageModel, dict[str, Any]]:
    """Restore a model and return checkpoint metadata."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    if not isinstance(config, ModelConfig):
        config = ModelConfig(**config)

    model = DecoderLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def load_training_state(
    path: str | Path,
    *,
    model: DecoderLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model and optimizer state into existing training objects."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
