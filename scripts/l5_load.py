import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch
from weird_ai.config import PROJECT_ROOT, SAMPLE_LYRICS_FILE
from weird_ai.model import WeirdAIModel
from weird_ai.tokenizer import SimpleCharacterTokenizer
from weird_ai.trainer import load_checkpoint
from weird_ai.generation import (
    text_to_token_ids, token_ids_to_text, generate_text_simple,
)


def main():
    device = torch.device("cuda")
    print("device:", device)

    corpus = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")
    tokenizer = SimpleCharacterTokenizer(corpus)
    vocab_size = len(tokenizer.chars)
    print("vocab_size:", vocab_size)

    model = WeirdAIModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    ckpt_path = PROJECT_ROOT / "models" / "lesson-05-pretrained" / "checkpoint.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    meta = load_checkpoint(model, optimizer, str(ckpt_path), device)

    print("\nRestored metadata is as follows:")
    print("epoch:", meta["epoch"])
    print("train_losses:", meta["train_losses"])
    print("val_losses:  ", meta["val_losses"])
    print("tokens_seen: ", meta["track_tokens_seen"])

    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mismatches = sum(
        1 for name, tensor in model.state_dict().items()
        if not torch.equal(tensor.cpu(), raw["model_state_dict"][name].cpu())
    )
    print(f"\nweight mismatch count: {mismatches} / {len(model.state_dict())}")

    context_size = 128
    print("\nGenerated samples:")
    for prompt in ["love is", "baby", "tonight"]:
        ids = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate_text_simple(
            model, ids, max_new_tokens=40, context_size=context_size
        )
        print(f"\nPrompt: {prompt!r}")
        print(token_ids_to_text(out, tokenizer))


if __name__ == "__main__":
    main()