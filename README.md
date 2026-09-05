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

## Train the language model

Training logs a progress bar with epoch, batch, loss, token throughput, ETA,
periodic validation loss, a final checkpoint, and by default one checkpoint per
completed epoch.

The training command defaults to `--device auto`, which chooses CUDA first, then
Apple MPS, then CPU. Pass `--device cuda`, `--device cuda:0`, `--device mps`, or
`--device cpu` to force a device. On CUDA machines, `--pin-memory` and
`--num-workers 4` can improve input-pipeline throughput.

On Apple Silicon, run training from your normal terminal rather than a
restricted sandbox so PyTorch can access Metal/MPS. The command logs `using
device: mps` when the M4 Pro GPU is active.

TinyStories is cached under `.hf-cache` after the first download, but Hugging
Face Datasets may still check the Hub for metadata. Add `--offline` after the
dataset exists locally to force cache-only loading. The training command also
caches encoded token IDs under `artifacts/datasets/`; later runs reuse that
cache and skip dataset iteration plus BPE encoding. Add
`--rebuild-encoded-cache` only when you intentionally want to regenerate it.

Quick smoke run:

```bash
HF_HOME=.hf-cache UV_CACHE_DIR=.uv-cache TIKTOKEN_CACHE_DIR=.tiktoken-cache \
uv run learning-llm-train \
  --max-documents 10000 \
  --context-length 64 \
  --d-model 128 \
  --n-head 4 \
  --n-layer 4 \
  --batch-size 32 \
  --epochs 1 \
  --max-steps 1000 \
  --eval-interval 100 \
  --device auto \
  --offline \
  --checkpoint artifacts/checkpoints/tinystories-tiny-lm.pt
```

Full TinyStories run with the default tiny architecture:

```bash
HF_HOME=.hf-cache UV_CACHE_DIR=.uv-cache TIKTOKEN_CACHE_DIR=.tiktoken-cache \
uv run learning-llm-train \
  --context-length 64 \
  --d-model 128 \
  --n-head 4 \
  --n-layer 4 \
  --batch-size 32 \
  --epochs 3 \
  --max-steps 50000 \
  --eval-interval 1000 \
  --device auto \
  --offline \
  --checkpoint artifacts/checkpoints/tinystories-tiny-lm.pt
```

Omitting `--max-documents` uses the full TinyStories split. Keep `--max-steps`
for bounded training; a full epoch over stride-1 windows is very large. For
example, 462M tokens with context length 128 and batch size 32 creates about
13M optimizer steps per epoch. At 3 steps/sec, three full epochs would take
around 149 days.

Full TinyStories run with GPT-2 Small architecture dimensions and the trained
50,000-token TinyStories BPE vocabulary:

```bash
HF_HOME=.hf-cache UV_CACHE_DIR=.uv-cache TIKTOKEN_CACHE_DIR=.tiktoken-cache \
uv run learning-llm-train \
  --model-preset gpt2-small \
  --batch-size 1 \
  --epochs 1 \
  --max-steps 10000 \
  --eval-interval 1000 \
  --device auto \
  --offline \
  --checkpoint artifacts/checkpoints/tinystories-gpt2-small-shape.pt
```

For intentional complete-epoch training, pass `--full-epochs --allow-long-run`.
Use `--stride 128` or another larger stride to reduce overlap between windows.
Use `--no-epoch-checkpoints` to save only the final checkpoint. Use
`--resume-from artifacts/checkpoints/tinystories-tiny-lm.pt` to continue from a
checkpoint without resetting optimizer state.

## Generate text

```bash
UV_CACHE_DIR=.uv-cache \
uv run learning-llm-generate \
  --checkpoint artifacts/checkpoints/tinystories-tiny-lm.pt \
  --tokenizer artifacts/tokenizers/tinystories-bpe-50k/tokenizer.json \
  --prompt "Once upon a time" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50
```

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
