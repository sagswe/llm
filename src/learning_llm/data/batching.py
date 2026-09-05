"""Small next-token batching utilities."""

from collections.abc import Iterator, Sequence

import torch


def next_token_windows(
    token_ids: Sequence[int],
    *,
    context_length: int,
    stride: int = 1,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield input/target windows for autoregressive training.

    For token IDs ``[a, b, c, d]`` and ``context_length=3``, the input is
    ``[a, b, c]`` and the target is ``[b, c, d]``. Each yielded tensor has shape
    ``(T,)``; batching stacks them into ``(B, T)`` later.
    """
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if len(token_ids) < context_length + 1:
        return

    for start in range(0, len(token_ids) - context_length, stride):
        end = start + context_length
        inputs = torch.tensor(token_ids[start:end], dtype=torch.long)
        targets = torch.tensor(token_ids[start + 1 : end + 1], dtype=torch.long)
        yield inputs, targets
