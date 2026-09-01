from torch.nn import LayerNorm, Module

from .FeedForward import FeedForward
from .MultiHeadSelfAttention import MultiHeadSelfAttention


class TransformerBlock(Module):

      def __init__(self, d_model, n_head, dropout=0.0):
          super().__init__();
          self.ln1 = LayerNorm(d_model)
          self.attn = MultiHeadSelfAttention(
              d_model,
              n_head,
              dropout,
          )

          self.ln2 = LayerNorm(d_model)
          self.ffn = FeedForward(
              d_model,
              mult=4,
              dropout=dropout,
          )

      def forward(self, x):
          attention_output, attention_weights = self.attn(
              self.ln1(x)
          )
          x = x + attention_output

          feed_forward_output = self.ffn(
              self.ln2(x)
          )
          x = x + feed_forward_output

          # Return attention weights for inspection alongside the transformed. this not needed for core transformer functionality, but can be useful for understanding what the model is attending to.
          # hidden states: x is (B, T, C), weights are (B, H, T, T).
          # only x is needed for the next transformer block, but we return the weights for inspection.
          return x, attention_weights
