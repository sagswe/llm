# Repository guidance

This repository is a teaching workspace for building a decoder-only language
model from scratch. Prefer code that exposes the underlying math and tensor
shapes. Do not optimize away an instructive step unless both the clear version
and the reason for the optimization remain easy to understand.

## Learning approach

- Build one small component at a time and accompany it with focused tests.
- Explain important tensor shapes in docstrings or nearby comments.
- Use equations and terminology consistently: batch `B`, sequence length `T`,
  model width `C`, number of heads `H`, and per-head width `D = C / H`.
- Prefer explicit PyTorch tensor operations over `torch.nn.Transformer`,
  `torch.nn.TransformerEncoder`, or `torch.nn.MultiheadAttention`.
- PyTorch autograd, basic layers such as `Linear`, `Embedding`, `LayerNorm`, and
  standard loss functions are allowed after their role is understood.
- Keep examples deterministic by seeding random number generators.
- Default tests and examples to tiny CPU-friendly dimensions.
- When adding a concept, state what it does, why it is needed, and how its output
  can be checked before composing it into a larger module.

## Branch progression

### `main`

Keep project configuration, shared documentation, and the learning roadmap here.

### `transformer`

Implement transformer internals in an educational order:

1. scaled dot-product attention;
2. causal masking;
3. single-head and multi-head self-attention;
4. feed-forward/MLP layer;
5. residual connections and layer normalization;
6. a decoder transformer block and a stack of blocks.

Keep tokenization, token embeddings, positional encoding/embeddings, language
model loss, training loops, and autoregressive generation out of this branch.

### `llm` (create later from `transformer`)

Assemble the complete decoder-only model. Add token and positional embeddings,
an LM head, weight tying where appropriate, training/evaluation loops,
checkpointing, and autoregressive text generation.

Use GPT-2 small as a conceptual reference: 12 blocks, 12 attention heads,
embedding width 768, context length 1024, and vocabulary size 50,257. Do not use
that full configuration as the default; provide tiny configurations for learning
and CPU execution.

## Verification

- Test tensor shapes, causal-mask behavior, numerical invariants, and gradients.
- Compare hand-written operations with a small trusted PyTorch equivalent when
  useful, without replacing the implementation under study.
- Run the relevant test suite through `uv run` before considering a component
  complete.
- Never silently fix a conceptual error: explain the misconception and make the
  smallest correction that demonstrates it.
