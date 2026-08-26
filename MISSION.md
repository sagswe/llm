# Mission: Learn LLMs from scratch

## Why

Develop a working mental model of modern language models by building a small,
GPT-2-style decoder-only model one understandable component at a time.

## Success looks like

- Explain the purpose, tensor shapes, and math of every model component.
- Implement and test transformer primitives without using black-box transformer APIs.
- Assemble, train, and sample from a small decoder-only language model.
- Relate the small implementation to GPT-2 small's architecture and parameters.

## Constraints

- Optimize for learning and clarity, not production performance.
- Use NumPy and PyTorch, with tiny CPU-friendly defaults.
- Introduce one concept at a time and verify it with focused tests.

## Out of scope

- Distributed training, production serving, and large-scale data pipelines.
- Reproducing GPT-2's trained weights or training cost.
