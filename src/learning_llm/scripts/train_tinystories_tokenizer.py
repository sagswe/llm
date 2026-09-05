"""Train a 50k byte-level BPE tokenizer on TinyStories."""

import argparse
from pathlib import Path

from learning_llm.data.tinystories import TinyStoriesConfig, iter_tinystories_texts
from learning_llm.data.tokenizers import train_byte_level_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tokenizers/tinystories-bpe-50k/tokenizer.json"),
        help="Path for the trained tokenizer JSON.",
    )
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Limit documents for quick experiments; omit for the full split.",
    )
    args = parser.parse_args()

    config = TinyStoriesConfig(
        split=args.split,
        streaming=args.streaming,
        max_documents=args.max_documents,
    )
    tokenizer = train_byte_level_bpe(
        iter_tinystories_texts(config),
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"saved tokenizer to {args.output}")
    print(f"vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
