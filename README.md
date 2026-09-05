# Learning LLM

Build a small GPT-style language model from first principles with NumPy and
PyTorch. The goal is understanding, so the code favors explicit tensor
operations and small, testable components over framework shortcuts.

## Setup

```bash
uv sync
uv run learning-llm
```

## Project layout

- `src/learning_llm/model`: transformer internals and the decoder backbone up
  to final layer normalization.
- `src/learning_llm/data`: tokenizer, TinyStories, and next-token batching
  utilities.
- `src/learning_llm/scripts`: command-line data preparation scripts.
- `examples`: standalone learning scripts for inspecting attention and linear
  layer dimensions.
- `artifacts/tokenizers/tinystories-bpe-50k/tokenizer.json`: trained
  TinyStories byte-level BPE tokenizer with a 50,000-token vocabulary.

Compatibility modules such as `learning_llm.TokenEmbedding` still import the
new implementations so earlier exercises continue to run.

## Train the TinyStories tokenizer

```bash
HF_HOME=.hf-cache UV_CACHE_DIR=.uv-cache \
uv run python -m learning_llm.scripts.train_tinystories_tokenizer \
  --vocab-size 50000 \
  --output artifacts/tokenizers/tinystories-bpe-50k/tokenizer.json
```

The default command downloads the TinyStories training split through Hugging
Face Datasets and trains on the full split. Add `--max-documents 10000` for a
quick local experiment.

## Branches

- `main`: project foundation and shared learning documentation.
- `transformer`: transformer primitives such as attention, MLPs, normalization,
  and a single transformer block.
- `llm` (later): the complete decoder-only language model, including token and
  positional embeddings, a stack of transformer blocks, training, and
  generation.

GPT-2 small (12 layers, 12 heads, 768-dimensional embeddings) is the reference
architecture. Exercises and tests should default to much smaller dimensions so
they run quickly on a CPU.
