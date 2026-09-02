"""Model-wide dimensions and settings for the decoder-only language model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Describe one decoder-only model without constructing any layers.

    The tiny defaults keep examples inexpensive on a CPU. Tensor dimensions use
    the repository's notation: model width ``C = d_model``, number of heads
    ``H = n_head``, and per-head width ``D = C / H``.
    """

    vocab_size: int = 32
    context_length: int = 16
    d_model: int = 8
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid dimensions before they reach a model component."""
        dimensions = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_head ({self.n_head})"
            )

        if (
            not isinstance(self.dropout, (int, float))
            or isinstance(self.dropout, bool)
            or not 0.0 <= self.dropout <= 1.0
        ):
            raise ValueError(
                f"dropout must be between 0.0 and 1.0, got {self.dropout!r}"
            )

    @property
    def head_dim(self) -> int:
        """Return per-head width ``D = C / H``."""
        return self.d_model // self.n_head

    @classmethod
    def gpt2_small(cls) -> "ModelConfig":
        """Return the GPT-2 Small reference dimensions.

        This preset is intentionally explicit and is not the default because it
        is too large for the repository's small CPU-friendly exercises.
        """
        return cls(
            vocab_size=50_257,
            context_length=1_024,
            d_model=768,
            n_head=12,
            n_layer=12,
            dropout=0.1,
        )
