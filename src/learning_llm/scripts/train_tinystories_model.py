"""Train a tiny decoder-only language model on TinyStories."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learning_llm.data import BPETokenizer, EncodedTextDataset, TinyStoriesConfig
from learning_llm.data.tinystories import iter_tinystories_texts
from learning_llm.model import DecoderLanguageModel, ModelConfig
from learning_llm.training import TrainingConfig, train


_TINY_PRESET = {
    "context_length": 64,
    "d_model": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.1,
}

_GPT2_SMALL_PRESET = {
    "context_length": 1_024,
    "d_model": 768,
    "n_head": 12,
    "n_layer": 12,
    "dropout": 0.1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizers/tinystories-bpe-50k/tokenizer.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/tinystories-tiny-lm.pt"),
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Limit documents for experiments; omit to train on the full split.",
    )
    parser.add_argument(
        "--model-preset",
        choices=("tiny", "gpt2-small"),
        default="tiny",
        help="Preset architecture. Explicit dimension flags override it.",
    )
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Distance between adjacent training windows in tokens.",
    )
    parser.add_argument(
        "--full-epochs",
        action="store_true",
        help="Ignore --max-steps and train for complete epochs.",
    )
    parser.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Allow planned runs above the safety threshold.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', 'cuda:0', or 'mps'.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin DataLoader memory when training on CUDA.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint to resume model and optimizer state from.",
    )
    parser.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help="Only save the final checkpoint path instead of one file per epoch.",
    )
    args = parser.parse_args()
    model_dimensions = _model_dimensions_from_args(args)

    torch.manual_seed(args.seed)
    tokenizer = BPETokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<eos>")

    print("loading TinyStories")
    text_config = TinyStoriesConfig(max_documents=args.max_documents)
    token_ids = _encode_documents(tokenizer, iter_tinystories_texts(text_config), eos_id)
    train_ids, validation_ids = _split_token_stream(
        token_ids,
        train_fraction=args.train_fraction,
        min_validation_tokens=model_dimensions["context_length"] + 1,
    )

    train_dataset = EncodedTextDataset(
        train_ids,
        context_length=model_dimensions["context_length"],
        stride=args.stride,
    )
    validation_dataset = EncodedTextDataset(
        validation_ids,
        context_length=model_dimensions["context_length"],
        stride=args.stride,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=model_dimensions["context_length"],
        d_model=model_dimensions["d_model"],
        n_head=model_dimensions["n_head"],
        n_layer=model_dimensions["n_layer"],
        dropout=model_dimensions["dropout"],
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=None if args.full_epochs else args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume_from,
        checkpoint_each_epoch=not args.no_epoch_checkpoints,
    )
    planned_steps = _planned_training_steps(
        train_windows=len(train_dataset),
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=training_config.max_steps,
    )
    _validate_run_size(planned_steps, allow_long_run=args.allow_long_run)

    print(
        "training setup "
        f"docs={'full' if args.max_documents is None else args.max_documents} "
        f"tokens={len(token_ids)} "
        f"train_windows={len(train_dataset)} "
        f"val_windows={len(validation_dataset)} "
        f"planned_steps={planned_steps} "
        f"vocab={tokenizer.vocab_size} "
        f"context={model_config.context_length} "
        f"stride={args.stride} "
        f"d_model={model_config.d_model} "
        f"layers={model_config.n_layer} "
        f"heads={model_config.n_head}"
    )
    model = DecoderLanguageModel(model_config)
    result = train(model, train_loader, validation_loader, training_config)
    print(
        "training complete "
        f"steps={result.steps} "
        f"last_train_loss={result.last_train_loss:.4f} "
        f"last_val_loss={result.last_validation_loss}"
    )


def _encode_documents(
    tokenizer: BPETokenizer,
    texts,
    eos_id: int | None,
) -> list[int]:
    token_stream = []
    for text in texts:
        token_stream.extend(tokenizer.encode(text))
        if eos_id is not None:
            token_stream.append(eos_id)
    return token_stream


def _model_dimensions_from_args(args: argparse.Namespace) -> dict[str, int | float]:
    preset = dict(_GPT2_SMALL_PRESET if args.model_preset == "gpt2-small" else _TINY_PRESET)
    overrides = {
        "context_length": args.context_length,
        "d_model": args.d_model,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "dropout": args.dropout,
    }
    for key, value in overrides.items():
        if value is not None:
            preset[key] = value
    return preset


def _planned_training_steps(
    *,
    train_windows: int,
    batch_size: int,
    epochs: int,
    max_steps: int | None,
) -> int:
    full_epoch_steps = train_windows // batch_size
    planned_steps = full_epoch_steps * epochs
    if max_steps is not None:
        planned_steps = min(planned_steps, max_steps)
    return planned_steps


def _validate_run_size(
    planned_steps: int,
    *,
    allow_long_run: bool,
    safety_threshold: int = 1_000_000,
) -> None:
    if planned_steps > safety_threshold and not allow_long_run:
        raise ValueError(
            "planned training run is very large: "
            f"{planned_steps:,} optimizer steps. "
            "Use --max-steps for a bounded run, increase --stride to sample "
            "fewer overlapping windows, or pass --allow-long-run if this is "
            "intentional."
        )


def _split_token_stream(
    token_ids: list[int],
    *,
    train_fraction: float,
    min_validation_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) < min_validation_tokens * 2:
        raise ValueError(
            "not enough tokens for train/validation windows; increase "
            "--max-documents or reduce --context-length"
        )
    split_index = int(len(token_ids) * train_fraction)
    split_index = min(split_index, len(token_ids) - min_validation_tokens)
    split_index = max(split_index, min_validation_tokens)
    tensor = torch.tensor(token_ids, dtype=torch.long)
    return tensor[:split_index], tensor[split_index:]


if __name__ == "__main__":
    main()
