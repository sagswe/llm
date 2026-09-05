"""Token and learned positional embeddings."""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Map token IDs to learned vectors.

    The embedding table has shape ``(V, C)`` where ``V`` is vocabulary size and
    ``C`` is model width. Input token IDs with shape ``(B, T)`` become token
    vectors with shape ``(B, T, C)``.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    @property
    def weight(self) -> torch.nn.Parameter:
        """Expose the table parameter for later LM-head weight tying."""
        return self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token vectors: ``(B, T) -> (B, T, C)``."""
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (B, T), got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"token_ids must contain integers, got {token_ids.dtype}")
        if token_ids.numel() > 0:
            min_id = int(token_ids.min().item())
            max_id = int(token_ids.max().item())
            if min_id < 0 or max_id >= self.vocab_size:
                raise ValueError(
                    f"token IDs must be in [0, {self.vocab_size}), "
                    f"got min={min_id}, max={max_id}"
                )

        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Add a learned vector for each token position.

    The embedding table has shape ``(context_length, C)``. For sequence length
    ``T``, rows ``0`` through ``T - 1`` form positional vectors with shape
    ``(T, C)``. PyTorch broadcasts them across batch ``B`` when added to token
    embeddings ``(B, T, C)``.
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

        position_ids = torch.arange(sequence_length, device=token_embeddings.device)
        position_vectors = self.embedding(position_ids)
        return token_embeddings + position_vectors
