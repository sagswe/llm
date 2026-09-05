"""Explicit causal self-attention components."""

import math

import torch
from torch.nn import Dropout, Linear, Module, ModuleList
from torch.nn.functional import softmax


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Mask future positions in attention scores.

    ``scores`` has shape ``(B, T, T)`` or ``(B, H, T, T)``. The returned tensor
    keeps the same shape but replaces entries above the main diagonal with
    ``-inf`` so softmax gives them probability zero.
    """
    if scores.ndim < 2:
        raise ValueError(f"scores must have at least 2 dims, got {scores.ndim}")

    sequence_length = scores.size(-1)
    if scores.size(-2) != sequence_length:
        raise ValueError(
            "attention score matrices must be square in the last two dims, "
            f"got {tuple(scores.shape)}"
        )

    causal_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            device=scores.device,
            dtype=torch.bool,
        )
    )
    return scores.masked_fill(~causal_mask, float("-inf"))


class SelfAttentionHead(Module):
    """Map model-width inputs to one causal attention head.

    Shape:
        ``(B, T, C) -> (B, T, D)``, where ``D`` is the per-head width.
    """

    def __init__(self, d_model: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.Wq = Linear(d_model, head_dim)
        self.Wk = Linear(d_model, head_dim)
        self.Wv = Linear(d_model, head_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return head output ``(B, T, D)`` and weights ``(B, T, T)``."""
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = apply_causal_mask(scores)

        weights = softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v
        return output, weights


class MultiHeadSelfAttention(Module):
    """Run ``H`` causal self-attention heads in parallel.

    Shapes:
        input: ``(B, T, C)``
        each head output: ``(B, T, D)``
        concatenated heads: ``(B, T, H * D) = (B, T, C)``
        attention weights: ``(B, H, T, T)``
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})"
            )

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.heads = ModuleList(
            [
                SelfAttentionHead(d_model, self.head_dim, dropout)
                for _ in range(n_head)
            ]
        )
        self.output_projection = Linear(d_model, d_model)
        self.output_dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixed output ``(B, T, C)`` and weights ``(B, H, T, T)``."""
        head_outputs = []
        head_attention_weights = []

        for head in self.heads:
            head_output, attention_weights = head(x)
            head_outputs.append(head_output)
            head_attention_weights.append(attention_weights)

        combined = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(combined)
        output = self.output_dropout(output)
        attention_weights = torch.stack(head_attention_weights, dim=1)
        return output, attention_weights
