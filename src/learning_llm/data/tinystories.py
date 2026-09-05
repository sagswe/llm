"""TinyStories dataset loading helpers."""

from collections.abc import Iterator
from dataclasses import dataclass

from datasets import DownloadConfig, load_dataset


@dataclass(frozen=True)
class TinyStoriesConfig:
    """Configuration for loading TinyStories text examples."""

    name: str = "roneneldan/TinyStories"
    split: str = "train"
    text_field: str = "text"
    streaming: bool = False
    max_documents: int | None = None
    local_files_only: bool = False


def iter_tinystories_texts(config: TinyStoriesConfig) -> Iterator[str]:
    """Yield TinyStories documents as plain text.

    With ``streaming=False``, Hugging Face Datasets downloads and caches the
    selected split before iteration. With ``streaming=True``, examples are read
    lazily without materializing the full split locally.
    """
    dataset = load_dataset(
        config.name,
        split=config.split,
        streaming=config.streaming,
        download_config=DownloadConfig(local_files_only=config.local_files_only),
    )

    count = 0
    for example in dataset:
        if config.max_documents is not None and count >= config.max_documents:
            break
        text = example[config.text_field]
        if text:
            yield text
            count += 1
