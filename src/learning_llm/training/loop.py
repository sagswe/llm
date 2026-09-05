"""Reusable training loop with progress bars and periodic evaluation."""

from dataclasses import dataclass
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from learning_llm.model import DecoderLanguageModel
from learning_llm.training.checkpoint import load_training_state, save_checkpoint
from learning_llm.training.config import TrainingConfig


@dataclass(frozen=True)
class EvaluationResult:
    """Average loss over a fixed number of validation batches."""

    loss: float
    batches: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run."""

    steps: int
    epochs: int
    last_train_loss: float
    last_validation_loss: float | None


def evaluate(
    model: DecoderLanguageModel,
    dataloader: DataLoader,
    *,
    device: str | torch.device,
    max_batches: int,
) -> EvaluationResult:
    """Measure mean next-token loss without tracking gradients."""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_index, (token_ids, targets) in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            token_ids = token_ids.to(device)
            targets = targets.to(device)
            output = model(token_ids, targets)
            losses.append(float(output.loss.item()))

    model.train()
    if not losses:
        raise ValueError("validation dataloader produced no batches")
    return EvaluationResult(loss=sum(losses) / len(losses), batches=len(losses))


def train(
    model: DecoderLanguageModel,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    config: TrainingConfig,
) -> TrainingResult:
    """Train with visible progress, ETA, loss, validation, and checkpoint logs."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    global_step = 0
    start_epoch = 1

    if config.resume_from is not None:
        checkpoint = load_training_state(
            config.resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        global_step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"])
        print(
            "resumed checkpoint "
            f"path={config.resume_from} "
            f"step={global_step} "
            f"epoch={start_epoch}"
        )

    batches_per_epoch = len(train_loader)
    planned_steps = batches_per_epoch * config.epochs
    if config.max_steps is not None:
        planned_steps = min(planned_steps, max(config.max_steps - global_step, 0))

    progress = tqdm(
        total=planned_steps,
        desc="training",
        unit="step",
        dynamic_ncols=True,
    )

    last_train_loss = float("nan")
    last_validation_loss = None
    start_time = perf_counter()
    epoch = start_epoch

    for epoch in range(start_epoch, config.epochs + 1):
        for batch_index, (token_ids, targets) in enumerate(train_loader, start=1):
            token_ids = token_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(token_ids, targets)
            if output.loss is None:
                raise RuntimeError("model did not return loss while targets were supplied")
            output.loss.backward()
            optimizer.step()

            global_step += 1
            last_train_loss = float(output.loss.item())
            elapsed = perf_counter() - start_time
            steps_per_second = global_step / max(elapsed, 1e-9)
            progress.set_postfix(
                {
                    "epoch": f"{epoch}/{config.epochs}",
                    "batch": f"{batch_index}/{batches_per_epoch}",
                    "train_loss": f"{last_train_loss:.4f}",
                    "tok/s": f"{token_ids.numel() * steps_per_second:.0f}",
                }
            )
            progress.update(1)

            if validation_loader is not None and global_step % config.eval_interval == 0:
                result = evaluate(
                    model,
                    validation_loader,
                    device=device,
                    max_batches=config.eval_batches,
                )
                last_validation_loss = result.loss
                progress.write(
                    "eval "
                    f"step={global_step} "
                    f"val_loss={result.loss:.4f} "
                    f"batches={result.batches}"
                )

            if config.max_steps is not None and global_step >= config.max_steps:
                break

        if config.max_steps is not None and global_step >= config.max_steps:
            break

    progress.close()
    save_checkpoint(
        config.checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=global_step,
        epoch=epoch,
        metadata={
            "last_train_loss": last_train_loss,
            "last_validation_loss": last_validation_loss,
        },
    )
    print(f"checkpoint saved: {config.checkpoint_path}")
    return TrainingResult(
        steps=global_step,
        epochs=epoch,
        last_train_loss=last_train_loss,
        last_validation_loss=last_validation_loss,
    )
