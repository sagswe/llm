# Learning LLM

Build a small GPT-style language model from first principles with NumPy and
PyTorch. The goal is understanding, so the code favors explicit tensor
operations and small, testable components over framework shortcuts.

## Setup

```bash
uv sync
uv run learning-llm
```

## Branches

- `main`: project foundation and shared learning documentation.
- `transformer`: transformer primitives such as attention, MLPs, normalization,
  and transformer blocks.
- `llm` (later): the complete decoder-only language model, including token and
  positional embeddings, training, and generation.

GPT-2 small (12 layers, 12 heads, 768-dimensional embeddings) is the reference
architecture. Exercises and tests should default to much smaller dimensions so
they run quickly on a CPU.
