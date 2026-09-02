"""GPT-2 tokenization through the tested ``tiktoken`` implementation."""

from collections.abc import Iterable

import tiktoken


class GPT2Tokenizer:
    """Convert text to GPT-2 token IDs and back.

    Tokenization produces a one-dimensional ``list[int]``. A later batching
    step will arrange token windows into PyTorch tensors with shape ``(B, T)``.
    """

    END_OF_TEXT = "<|endoftext|>"

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        """Return GPT-2's vocabulary size ``V = 50_257``."""
        return self._encoding.n_vocab

    @property
    def end_of_text_id(self) -> int:
        """Return the token ID used to mark the end of a document."""
        return self._encoding.eot_token

    def encode(self, text: str, *, add_end_of_text: bool = False) -> list[int]:
        """Encode text and optionally append GPT-2's end-of-text token.

        A literal ``<|endoftext|>`` in ordinary input is rejected by tiktoken.
        This keeps special-token insertion explicit through ``add_end_of_text``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        token_ids = self._encoding.encode(text)
        if add_end_of_text:
            token_ids.append(self.end_of_text_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode a one-dimensional iterable of GPT-2 token IDs into text."""
        ids = list(token_ids)
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(
                    "each token ID must be an integer, "
                    f"got {token_id!r}"
                )
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token ID must be in [0, {self.vocab_size}), got {token_id}"
                )

        return self._encoding.decode(ids)
