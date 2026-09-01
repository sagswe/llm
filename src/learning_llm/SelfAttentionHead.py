import math

import torch
from torch.nn import Dropout, Linear, Module

from torch.nn.functional import softmax


def apply_causal_mask(scores):
    """Mask out future positions so each token can only attend to past tokens."""
    seq_length = scores.size(-1)
    causal_mask = torch.tril(
        torch.ones(seq_length, seq_length, device=scores.device, dtype=torch.bool)
    )
    return scores.masked_fill(~causal_mask, float("-inf"))


class SelfAttentionHead(Module):
    """Map model-width inputs to one causal attention head's output.

    Shape:
        (batch_size, sequence_length, model_width)
        -> (batch_size, sequence_length, head_width)
    """

    def __init__(self, d_model, head_dim, dropout=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.Wq = Linear(d_model, head_dim)
        self.Wk = Linear(d_model, head_dim)
        self.Wv = Linear(d_model, head_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, sequence_length, model_width)
        # q, k, v: (batch_size, sequence_length, head_width)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # Compare every query with every key:
        # (batch_size, sequence_length, sequence_length).
        scores = (q @ self.transpose_last_two_dims(k)) / math.sqrt(self.head_dim)
        scores = apply_causal_mask(scores)

        weights = softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values:
        # (batch_size, sequence_length, sequence_length)
        # @ (batch_size, sequence_length, head_width)
        # -> (batch_size, sequence_length, head_width).
        output = weights @ v
        return output, weights

    def transpose_last_two_dims(self, tensor):
        """Transpose the last two dimensions of a tensor."""
        return tensor.transpose(-1, -2)
