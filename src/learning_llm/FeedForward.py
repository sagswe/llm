from torch.nn import Dropout, GELU, Linear, Module


class FeedForward(Module):
    """Apply a position-wise MLP without changing the tensor's outer shape.

    The same two-layer network is applied independently to every token:

        (B, T, C) -> Linear(C, mult * C) -> GELU
                  -> Linear(mult * C, C) -> Dropout -> (B, T, C)

    Expanding the feature dimension gives each token more capacity to transform
    the information gathered by attention. The final projection returns to C so
    the result can be added to the residual stream.
    """

    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        hidden_dim = mult * d_model

        self.input_projection = Linear(d_model, hidden_dim)
        self.activation = GELU()
        self.output_projection = Linear(hidden_dim, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # x:      (B, T, C)
        # hidden: (B, T, mult * C)
        hidden = self.input_projection(x)
        hidden = self.activation(hidden)

        # Return to model width so the transformer block can add its residual.
        output = self.output_projection(hidden)  # (B, T, C)
        return self.dropout(output)
