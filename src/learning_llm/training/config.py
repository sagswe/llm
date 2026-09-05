"""Training configuration for tiny CPU-friendly runs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for an autoregressive language-model training run."""

    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int | None = 200
    eval_interval: int = 50
    eval_batches: int = 10
    seed: int = 1337
    device: str = "cpu"
    checkpoint_path: Path = Path("artifacts/checkpoints/tiny-lm.pt")
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}")
        if self.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {self.eval_interval}")
        if self.eval_batches <= 0:
            raise ValueError(f"eval_batches must be positive, got {self.eval_batches}")
