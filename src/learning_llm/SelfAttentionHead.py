"""Compatibility imports for attention heads."""

from learning_llm.model.attention import SelfAttentionHead, apply_causal_mask

__all__ = ["SelfAttentionHead", "apply_causal_mask"]
