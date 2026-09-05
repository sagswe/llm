"""Language-model projection from hidden states to vocabulary logits."""

from dataclasses import dataclass

import torch
from torch import nn

from learning_llm.model.backbone import DecoderBackbone
from learning_llm.model.config import ModelConfig


class LanguageModelHead(nn.Module):
    """Project hidden states to next-token scores.

    The projection maps each normalized hidden vector from width ``C`` to one
    unnormalized score per vocabulary token ``V``:
    ``(B, T, C) -> (B, T, V)``.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def weight(self) -> torch.nn.Parameter:
        """Expose projection weights for checking weight tying."""
        return self.projection.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return vocabulary logits with shape ``(B, T, V)``."""
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape (B, T, C), "
                f"got {tuple(hidden_states.shape)}"
            )
        return self.projection(hidden_states)


@dataclass(frozen=True)
class DecoderLanguageModelOutput:
    """Output from the decoder language model before adding training loss."""

    logits: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...] | None = None


class DecoderLanguageModel(nn.Module):
    """Decoder backbone plus tied language-model head.

    This completes the inference path through Step 7. Loss computation remains
    separate for the next milestone.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = DecoderBackbone(config)
        self.lm_head = LanguageModelHead(config.d_model, config.vocab_size)
        self.tie_weights()

    def tie_weights(self) -> None:
        """Share one parameter between input embeddings and output projection."""
        self.lm_head.projection.weight = self.backbone.token_embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> DecoderLanguageModelOutput:
        """Return logits ``(B, T, V)`` for next-token prediction."""
        backbone_output = self.backbone(
            token_ids,
            return_attention_weights=return_attention_weights,
        )
        logits = self.lm_head(backbone_output.hidden_states)
        return DecoderLanguageModelOutput(
            logits=logits,
            attention_weights=backbone_output.attention_weights,
        )
