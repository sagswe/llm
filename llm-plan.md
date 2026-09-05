# Decoder-Only LLM Implementation Plan

This document tracks the work required to turn the transformer components into
a complete decoder-only language model. The implementation should remain
educational: use small deterministic examples, expose important tensor shapes,
and introduce one concept at a time before composing the complete model.

The transformer branch already provides causal self-attention, multi-head
self-attention, a feed-forward layer, residual connections, layer
normalization, and a single transformer block.

## Reference configuration

GPT-2 Small is the conceptual reference configuration:

- Number of transformer blocks: `12`
- Number of attention heads: `12`
- Model width: `C = 768`
- Per-head width: `D = C / H = 64`
- Maximum context length: `1,024`
- Vocabulary size: `50,257`

Do not use the full GPT-2 Small configuration as the default. Examples should
use tiny CPU-friendly values, such as two blocks, two heads, model width eight,
context length sixteen, and a small vocabulary.

## Progress checklist

### 1. Model configuration

- [x] Define a configuration object for model dimensions and behavior.
- [x] Provide a tiny default configuration for examples and development.
- [x] Provide an explicit GPT-2 Small reference configuration.
- [x] Validate invariants such as `C % H == 0` and `T <= context_length`.

The configuration should be the single source of truth for vocabulary size,
context length, model width, number of heads, number of blocks, dropout, and
related settings. Modules should not repeat hard-coded GPT-2 dimensions.

Example scenario: construct a tiny model with `vocab_size=32`,
`context_length=16`, `d_model=8`, `n_head=2`, and `n_layer=2`. Attempting to use
three heads with model width eight should produce a clear validation error.

### 2. Tokenizer

- [x] Define a small tokenizer interface with `encode` and `decode` operations.
- [x] Add a simple tokenizer suitable for learning and tiny local examples.
- [x] Add GPT-2-compatible byte-level BPE tokenization when compatibility is
      needed.
- [x] Handle the end-of-text or other explicitly supported special tokens.
- [x] Confirm that representative text survives an encode/decode round trip.

Tokenization converts text into integer token IDs and back. Keeping the model
independent from the concrete tokenizer will let us first learn with a small
vocabulary and later use GPT-2's vocabulary of 50,257 tokens.

Example scenario: encoding `"hello world"` returns a one-dimensional sequence
of integer IDs. Decoding those IDs reconstructs the original text. A batch is
formed later by arranging token windows into a tensor of shape `(B, T)`.

### 3. Token embeddings

- [x] Map token IDs `(B, T)` to token vectors `(B, T, C)`.
- [x] Document how each embedding row corresponds to one vocabulary token.
- [x] Reject or clearly report token IDs outside `[0, V)`.

An embedding table has shape `(V, C)`, where `V` is vocabulary size. Looking up
each input ID produces the initial learned representation of that token.

Example scenario: input IDs with shape `(2, 5)` and model width eight produce
token embeddings with shape `(2, 5, 8)`.

### 4. Learned positional embeddings

- [x] Create learned embeddings for positions `0` through
      `context_length - 1`.
- [x] Add positional embeddings to token embeddings without changing shape.
- [x] Validate that the requested sequence fits inside the context length.

Causal attention knows which tokens are visible but does not otherwise know
their positions. Learned positional embeddings give the model a distinct vector
for each position. Token and positional embeddings both have shape `(B, T, C)`
after broadcasting and can therefore be added.

Example scenario: the same token appearing at positions two and seven begins
with the same token vector but receives different positional information.

### 5. Stack of transformer blocks

- [x] Build a module containing `N` transformer blocks.
- [x] Pass hidden states through each block in order.
- [x] Keep the hidden-state shape `(B, T, C)` unchanged across the stack.
- [x] Optionally collect attention weights for inspection without coupling core
      model behavior to visualization.

Each block refines the residual stream. Stacking belongs to the complete LLM
rather than the standalone transformer primitive because the number of layers
is a model-level configuration choice.

Example scenario: a two-block tiny model accepts `(2, 5, 8)` hidden states and
returns `(2, 5, 8)`. If attention weights are requested, each block exposes
weights shaped `(B, H, T, T)`.

### 6. Final layer normalization

- [x] Apply a final layer normalization after the transformer stack.
- [x] Confirm that normalization preserves `(B, T, C)`.

GPT-style pre-normalized blocks normalize before each sublayer and also apply a
final normalization before producing vocabulary logits.

### 7. Language-model head and weight tying

- [x] Project normalized hidden states `(B, T, C)` to logits `(B, T, V)`.
- [x] Tie the language-model head weight to the token embedding weight.
- [x] Verify that both modules reference the same parameter rather than copied
      values.

The logit at `(b, t, v)` is the model's unnormalized score for vocabulary token
`v` following position `t`. Weight tying reduces parameter count and lets input
and output token representations share a learned coordinate system.

Example scenario: with vocabulary size 32, hidden states `(2, 5, 8)` produce
logits `(2, 5, 32)`. Updating the shared weight through either use must update a
single parameter.

### 8. Complete decoder-only model

- [x] Compose token embeddings, positional embeddings, dropout, transformer
      blocks, final normalization, and the LM head.
- [x] Keep the forward path readable and annotate all important shapes.
- [x] Support inference without requiring targets.
- [x] Initialize parameters deliberately and document the chosen scheme.

The model forward pass should follow this sequence:

1. Convert `(B, T)` token IDs into `(B, T, C)` token embeddings.
2. Add position embeddings with compatible shape.
3. Pass the residual stream through `N` transformer blocks.
4. Apply final layer normalization.
5. Project to vocabulary logits `(B, T, V)`.

### 9. Next-token language-model loss

- [x] Accept target token IDs with shape `(B, T)` when training.
- [x] Compute cross-entropy between logits and next-token targets.
- [x] Reshape logits and targets explicitly to show how token positions become
      independent classification examples.
- [x] Support an ignore index if padded or intentionally masked targets are
      introduced.

For an autoregressive batch, inputs and targets are shifted views of the same
token stream. For tokens `[a, b, c, d]`, input `[a, b, c]` has target
`[b, c, d]`. Logits `(B, T, V)` are flattened to `(B*T, V)` and targets to
`(B*T)` before cross-entropy.

Example scenario: a model given `"the cat sat"` learns that after the prefix
`"the cat"`, the next target token is part of `" sat"`.

### 10. Dataset and batching utilities

- [x] Load a small text corpus deterministically.
- [x] Tokenize the corpus once or document when tokenization occurs.
- [x] Create fixed-length input and next-token target windows.
- [x] Batch windows into tensors shaped `(B, T)`.
- [x] Separate training and validation data without leakage.

Example scenario: for token IDs `[0, 1, 2, 3, 4, 5]` and context length three,
one sample can use input `[0, 1, 2]` and target `[1, 2, 3]`. Sampling policy
should state whether windows overlap and whether starts are sequential or
random.

### 11. Training and evaluation loops

- [x] Seed all relevant random number generators.
- [x] Implement a clear training step: zero gradients, forward pass, loss,
      backward pass, and optimizer update.
- [x] Report training and validation losses at controlled intervals.
- [x] Use evaluation mode and disable gradient tracking during evaluation.
- [x] Keep default runs small enough for a CPU.

Example scenario: deliberately train a tiny model on a very small repeated
sequence and confirm that it can overfit it. This is a useful end-to-end check
before training on a larger corpus.

### 12. Checkpointing

- [x] Save model parameters, optimizer state, configuration, training step, and
      relevant metadata.
- [x] Restore a checkpoint onto an explicitly selected device.
- [x] Resume training without silently resetting optimizer progress.
- [x] Confirm a save/load round trip produces identical logits in evaluation
      mode.

Checkpoints should contain enough information to reconstruct the model without
depending on undocumented command-line defaults.

### 13. Autoregressive text generation

- [x] Implement greedy next-token selection.
- [x] Add temperature-controlled sampling.
- [x] Add top-k sampling.
- [x] Crop long input histories to the configured context length.
- [x] Stop at an end-of-text token when configured to do so.
- [x] Preserve deterministic behavior when a seed is supplied.

At each generation step, run the model on the available context, take logits
from the final time position `(B, V)`, select or sample the next token, append
it, and repeat.

Example scenarios:

- Temperature zero or an explicit greedy mode always selects the highest-logit
  token.
- A higher temperature flattens the probability distribution and increases
  sampling diversity.
- Top-k sampling with `k=5` prevents selection outside the five highest-scoring
  candidates.
- Once a prompt exceeds the context length, only its most recent tokens are
  passed to the model.

### 14. Focused verification

- [ ] Check tensor shapes at every model boundary.
- [ ] Check context-length and configuration validation.
- [ ] Check tokenization round trips.
- [ ] Check that weight tying uses one shared parameter.
- [ ] Check finite logits, loss, and gradients.
- [ ] Check that a training step can reduce loss on a tiny fixed batch.
- [ ] Check deterministic generation with a fixed seed.
- [ ] Check checkpoint save/load equivalence.
- [ ] Compare a small hand-written calculation with a trusted PyTorch operation
      where it clarifies the math.

Verification can initially use small executable examples if formal tests are
being deferred, but each check should be repeatable and fail loudly when its
invariant is broken.

### 15. Optional GPT-2 compatibility

- [ ] Match GPT-2 module dimensions, parameter names, biases, activations, and
      normalization details where necessary.
- [ ] Load GPT-2 Small pretrained weights into the from-scratch model.
- [ ] Verify tokenization and selected logits against a trusted GPT-2
      implementation.
- [ ] Generate text from pretrained weights.

This milestone should begin only after the tiny from-scratch model trains and
generates correctly. Compatibility work must not obscure the underlying model
implementation.

## Suggested implementation order

Complete one focused milestone at a time:

1. Configuration
2. Simple tokenizer
3. Token and positional embeddings
4. Transformer stack and final normalization
5. LM head and weight tying
6. Complete forward pass
7. Next-token loss
8. Dataset and batching
9. Training and evaluation
10. Checkpointing
11. Generation
12. GPT-2 tokenizer and pretrained-weight compatibility, if desired

Update each checkbox as its implementation and repeatable verification are
completed. Avoid marking a composed feature complete merely because its
individual modules exist; verify the data and shapes at their boundary first.
