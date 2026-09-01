import torch
from torch.nn import Dropout, Linear, Module, ModuleList

from .SelfAttentionHead import SelfAttentionHead


class MultiHeadSelfAttention(Module):
    """Run n_head self-attention heads in parallel and combine their results.

    Shapes:
        input x:            (batch_size, seq_length, d_model)
        each head output:   (batch_size, seq_length, head_dim), where head_dim = d_model / n_head
        concatenated heads: (batch_size, seq_length, n_head * head_dim) = (batch_size, seq_length, d_model)
        attention weights:  (batch_size, n_head, seq_length, seq_length)

    """

    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__();

        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})"
            )
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.heads = ModuleList(
            [
                SelfAttentionHead(
                    d_model,
                    self.head_dim,
                    dropout,
                )
                for _ in range(n_head)
            ]
        )

        # Mix information from all H heads after concatenation.
        self.output_projection = Linear(d_model, d_model)
        self.output_dropout = Dropout(dropout)

    def forward(self, x):
        # x has shape (batch_size, seq_length, d_model).
        head_outputs =  list()
        head_attention_weights =  list()

        for head in self.heads:
            head_output, attention_weights = head(x)

            # head_output:       (batch_size, seq_length, head_dim)
            # attention_weights: (batch_size, seq_length, seq_length)
            head_outputs.append(head_output)
            head_attention_weights.append(attention_weights)

        # Join the n_head head outputs along their feature dimension:
        # n_head tensors of (batch_size, seq_length, head_dim) -> (batch_size, seq_length, n_head * head_dim) = (batch_size, seq_length, d_model).
        combined = torch.cat(head_outputs, dim=-1)

        output = self.output_projection(combined)
        output = self.output_dropout(output)

        # Keep the head dimension visible for inspection:
        # n_head tensors of (batch_size, seq_length, seq_length)
        # -> (batch_size, n_head, seq_length, seq_length).
        attention_weights = torch.stack(
            head_attention_weights,
            dim=1,
        )

        return output, attention_weights
