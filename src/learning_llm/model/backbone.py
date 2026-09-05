"""Decoder-only backbone up to final normalization, before the LM head."""

from dataclasses import dataclass

import torch
from torch import nn

from learning_llm.model.blocks import TransformerStack
from learning_llm.model.config import ModelConfig
from learning_llm.model.embeddings import PositionalEmbedding, TokenEmbedding


@dataclass(frozen=True)
class DecoderBackboneOutput:
    """Output from the model backbone before vocabulary projection."""

    hidden_states: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...] | None = None


class DecoderBackbone(nn.Module):
    """Compose embeddings, decoder blocks, and final normalization.

    This intentionally stops before the language-model head. The forward path:

    1. token IDs ``(B, T)`` -> token embeddings ``(B, T, C)``
    2. add learned positions, preserving ``(B, T, C)``
    3. pass through ``N`` transformer blocks, preserving ``(B, T, C)``
    4. final layer norm, preserving ``(B, T, C)``
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.position_embedding = PositionalEmbedding(
            config.context_length,
            config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = TransformerStack(
            n_layer=config.n_layer,
            d_model=config.d_model,
            n_head=config.n_head,
            dropout=config.dropout,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.apply(self._init_parameters)

    def _init_parameters(self, module: nn.Module) -> None:
        """Use GPT-style small normal initialization for learned weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> DecoderBackboneOutput:
        """Return normalized hidden states with shape ``(B, T, C)``."""
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (B, T), got {tuple(token_ids.shape)}"
            )

        _, sequence_length = token_ids.shape
        self.config.validate_sequence_length(sequence_length)

        hidden_states = self.token_embedding(token_ids)
        hidden_states = self.position_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, attention_weights = self.blocks(
            hidden_states,
            return_attention_weights=return_attention_weights,
        )
        hidden_states = self.final_norm(hidden_states)

        return DecoderBackboneOutput(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
        )
