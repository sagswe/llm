"""Educational decoder-only language model package."""

from learning_llm.model import (
    DecoderBackbone,
    DecoderBackboneOutput,
    ModelConfig,
    TransformerBlock,
    TransformerStack,
)

__all__ = [
    "DecoderBackbone",
    "DecoderBackboneOutput",
    "ModelConfig",
    "TransformerBlock",
    "TransformerStack",
    "main",
]


def main() -> None:
    print("learning-llm package is installed")
