"""Tokenizer interfaces and trainable byte-level BPE tokenizer."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface used by data utilities and examples."""

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size ``V``."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back into text."""


class CharacterTokenizer:
    """Small deterministic tokenizer for toy examples.

    It assigns one token ID to each character in a provided vocabulary. Encoding
    text returns a one-dimensional ``list[int]``; later data utilities arrange
    windows into tensors with shape ``(B, T)``.
    """

    def __init__(self, vocabulary: Iterable[str], unk_token: str = "<unk>"):
        tokens = [unk_token, *dict.fromkeys(vocabulary)]
        if any(len(token) != 1 and token != unk_token for token in tokens):
            raise ValueError("character vocabulary entries must be single characters")

        self.unk_token = unk_token
        self.id_to_token = list(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size ``V``."""
        return len(self.id_to_token)

    @classmethod
    def from_text(cls, text: str, unk_token: str = "<unk>") -> "CharacterTokenizer":
        """Build a sorted character vocabulary from local text."""
        return cls(sorted(set(text)), unk_token=unk_token)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        unk_id = self.token_to_id[self.unk_token]
        return [self.token_to_id.get(char, unk_id) for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back to text, replacing unknown IDs with ``<unk>``."""
        chars = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"token IDs must be integers, got {token_id!r}")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token ID must be in [0, {self.vocab_size}), got {token_id}"
                )
            chars.append(self.id_to_token[token_id])
        return "".join(chars)


class BPETokenizer:
    """Wrapper around a trained Hugging Face `tokenizers` byte-level BPE file."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer JSON file produced by `train_byte_level_bpe`."""
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size ``V``."""
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        return self.tokenizer.encode(text).ids

    def token_to_id(self, token: str) -> int | None:
        """Return the ID for a special or vocabulary token when present."""
        return self.tokenizer.token_to_id(token)

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode token IDs into text."""
        ids = list(token_ids)
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"token IDs must be integers, got {token_id!r}")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"token ID must be in [0, {self.vocab_size}), got {token_id}"
                )
        return self.tokenizer.decode(ids)


def train_byte_level_bpe(
    texts: Iterable[str],
    output_path: str | Path,
    *,
    vocab_size: int = 50_000,
    min_frequency: int = 2,
    special_tokens: tuple[str, ...] = ("<unk>", "<pad>", "<bos>", "<eos>"),
) -> BPETokenizer:
    """Train and save a byte-level BPE tokenizer from text examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_non_empty_texts(texts), trainer=trainer)
    tokenizer.save(str(output_path))
    return BPETokenizer(tokenizer)


def _non_empty_texts(texts: Iterable[str]) -> Iterator[str]:
    for text in texts:
        if not isinstance(text, str):
            raise TypeError(f"training texts must be strings, got {type(text).__name__}")
        if text:
            yield text
