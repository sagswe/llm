import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Add a learned vector for each token position.

    The embedding table has shape ``(context_length, d_model)``. For an input
    sequence of length ``T``, rows ``0`` through ``T - 1`` are selected to form
    positional vectors with shape ``(T, C)``. PyTorch broadcasts those vectors
    across the batch when they are added to token embeddings ``(B, T, C)``.
    """

    def __init__(self, context_length: int, d_model: int):
        super().__init__()
        if context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {context_length}"
            )
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.context_length = context_length
        self.d_model = d_model
        self.embedding = nn.Embedding(context_length, d_model)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Add positions to token embeddings: ``(B, T, C) -> (B, T, C)``."""
        if token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must have shape (B, T, C), "
                f"got {tuple(token_embeddings.shape)}"
            )

        _, sequence_length, model_width = token_embeddings.shape
        if model_width != self.d_model:
            raise ValueError(
                f"input model width C ({model_width}) must equal "
                f"d_model ({self.d_model})"
            )
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length T ({sequence_length}) exceeds "
                f"context_length ({self.context_length})"
            )

        # position_ids: (T,)
        # position_vectors: (T, C), broadcast over B during addition.
        position_ids = torch.arange(
            sequence_length,
            device=token_embeddings.device,
        )
        position_vectors = self.embedding(position_ids)
        return token_embeddings + position_vectors
