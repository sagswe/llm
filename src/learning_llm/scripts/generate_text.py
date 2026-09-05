"""Generate text from a trained decoder language-model checkpoint."""

import argparse
from pathlib import Path

import torch

from learning_llm.data import BPETokenizer
from learning_llm.training import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/tinystories-tiny-lm.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizers/tinystories-bpe-50k/tokenizer.json"),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', 'cuda:0', or 'mps'.",
    )
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_file(args.tokenizer)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    prompt_ids = prompt_ids.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    eos_id = tokenizer.token_to_id("<eos>")

    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_id,
        generator=generator,
    )
    text = tokenizer.decode(output_ids[0].tolist())
    print(f"loaded checkpoint step={checkpoint.get('step')} epoch={checkpoint.get('epoch')}")
    print(text)


if __name__ == "__main__":
    main()
