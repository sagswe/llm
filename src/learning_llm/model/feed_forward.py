"""Position-wise feed-forward network used inside decoder blocks."""

from torch import Tensor
from torch.nn import Dropout, GELU, Linear, Module


class FeedForward(Module):
    """Apply a two-layer MLP independently to every token.

    Shape:
        ``(B, T, C) -> Linear(C, 4C) -> GELU -> Linear(4C, C) -> (B, T, C)``
    """

    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if mult <= 0:
            raise ValueError(f"mult must be positive, got {mult}")

        hidden_dim = mult * d_model
        self.input_projection = Linear(d_model, hidden_dim)
        self.activation = GELU()
        self.output_projection = Linear(hidden_dim, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Preserve hidden-state shape ``(B, T, C)``."""
        hidden = self.input_projection(x)
        hidden = self.activation(hidden)
        output = self.output_projection(hidden)
        return self.dropout(output)
