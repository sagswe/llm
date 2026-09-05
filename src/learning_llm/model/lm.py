"""Language-model projection from hidden states to vocabulary logits."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F
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
    """Output from the decoder language model.

    ``logits`` has shape ``(B, T, V)``. During training, ``loss`` is computed by
    flattening logits to ``(B*T, V)`` and targets to ``(B*T)`` so every token
    position becomes one vocabulary-classification example.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    attention_weights: tuple[torch.Tensor, ...] | None = None


class DecoderLanguageModel(nn.Module):
    """Decoder backbone plus tied language-model head.

    The model accepts token IDs ``(B, T)`` and returns logits ``(B, T, V)``.
    When next-token targets ``(B, T)`` are supplied, it also returns
    cross-entropy loss.
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
        targets: torch.Tensor | None = None,
        *,
        ignore_index: int = -100,
        return_attention_weights: bool = False,
    ) -> DecoderLanguageModelOutput:
        """Return logits and optional next-token loss.

        Shape path:
            token IDs: ``(B, T)``
            hidden states: ``(B, T, C)``
            logits: ``(B, T, V)``
            flattened logits for loss: ``(B*T, V)``
        """
        backbone_output = self.backbone(
            token_ids,
            return_attention_weights=return_attention_weights,
        )
        logits = self.lm_head(backbone_output.hidden_states)
        loss = None

        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError(
                    "targets must have the same shape as token_ids, "
                    f"got targets={tuple(targets.shape)} and "
                    f"token_ids={tuple(token_ids.shape)}"
                )
            logits_by_position = logits.reshape(-1, self.config.vocab_size)
            targets_by_position = targets.reshape(-1)
            loss = F.cross_entropy(
                logits_by_position,
                targets_by_position,
                ignore_index=ignore_index,
            )

        return DecoderLanguageModelOutput(
            logits=logits,
            loss=loss,
            attention_weights=backbone_output.attention_weights,
        )

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively append tokens to ``token_ids``.

        At each step, only the most recent ``context_length`` tokens are passed
        through the model. ``temperature=0`` performs greedy decoding; positive
        temperatures sample from the softmax distribution.
        """
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (B, T), got {tuple(token_ids.shape)}"
            )
        if max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be non-negative, got {max_new_tokens}"
            )
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive when set, got {top_k}")

        self.eval()
        generated = token_ids

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.context_length :]
            logits = self(context).logits[:, -1, :]
            next_token = _select_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return generated


def _select_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Select one token per batch row from final-position logits ``(B, V)``."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    if top_k is not None:
        kept = min(top_k, scaled_logits.size(-1))
        values, _ = torch.topk(scaled_logits, kept, dim=-1)
        threshold = values[:, [-1]]
        scaled_logits = scaled_logits.masked_fill(
            scaled_logits < threshold,
            float("-inf"),
        )

    probabilities = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
