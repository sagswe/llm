"""Decoder transformer blocks and stacks."""

import torch
from torch.nn import LayerNorm, Module, ModuleList

from learning_llm.model.attention import MultiHeadSelfAttention
from learning_llm.model.feed_forward import FeedForward


class TransformerBlock(Module):
    """One pre-norm decoder block.

    Hidden states keep shape ``(B, T, C)`` through attention, residual addition,
    feed-forward transformation, and the second residual addition.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mult=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hidden states ``(B, T, C)`` and weights ``(B, H, T, T)``."""
        attention_output, attention_weights = self.attn(self.ln1(x))
        x = x + attention_output

        feed_forward_output = self.ffn(self.ln2(x))
        x = x + feed_forward_output
        return x, attention_weights


class TransformerStack(Module):
    """Apply ``N`` decoder blocks in order without changing shape."""

    def __init__(
        self,
        n_layer: int,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {n_layer}")

        self.blocks = ModuleList(
            [
                TransformerBlock(d_model=d_model, n_head=n_head, dropout=dropout)
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Run blocks over hidden states ``(B, T, C)``."""
        all_attention_weights = []

        for block in self.blocks:
            x, attention_weights = block(x)
            if return_attention_weights:
                all_attention_weights.append(attention_weights)

        if return_attention_weights:
            return x, tuple(all_attention_weights)
        return x, None
