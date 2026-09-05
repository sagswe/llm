"""Dataset helpers for next-token language modeling."""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EncodedTextDataset(Dataset):
    """Fixed windows from a single token stream.

    If the encoded stream is ``[0, 1, 2, 3]`` and context length is ``3``, the
    first item returns input ``[0, 1, 2]`` and target ``[1, 2, 3]``. Every item
    returns tensors with shape ``(T,)``; a DataLoader stacks them into ``(B, T)``.
    """

    token_ids: torch.Tensor
    context_length: int
    stride: int = 1

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must have shape (N,), got {tuple(self.token_ids.shape)}"
            )
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.token_ids.numel() < self.context_length + 1:
            raise ValueError(
                "token_ids must contain at least context_length + 1 IDs, "
                f"got {self.token_ids.numel()}"
            )

    def __len__(self) -> int:
        usable_starts = self.token_ids.numel() - self.context_length
        return (usable_starts + self.stride - 1) // self.stride

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.stride
        end = start + self.context_length
        inputs = self.token_ids[start:end].long()
        targets = self.token_ids[start + 1 : end + 1].long()
        return inputs, targets


def split_token_ids(
    token_ids: list[int],
    *,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one encoded stream into train/validation token tensors."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    if len(token_ids) < 2:
        raise ValueError("token_ids must contain at least two IDs")

    split_index = max(1, min(len(token_ids) - 1, int(len(token_ids) * train_fraction)))
    tensor = torch.tensor(token_ids, dtype=torch.long)
    return tensor[:split_index], tensor[split_index:]
