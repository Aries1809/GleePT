#!/usr/bin/env python3

from pathlib import Path
import tokenizer

DATA_PATH = Path(__file__).parent.parent / "input.txt"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def main() -> None:
    text = load_text(DATA_PATH)
    stoi, itos = build_vocab(text)

    print(text[:500])
    print(f"\nlength:     {len(text)} characters")
    print(f"vocab:      {''.join(itos.values())!r}")
    print(f"vocab size: {len(itos)}")

    sample = "hello there"
    tokens = tokenizer.encode(sample, stoi)
    print(f"{sample!r} -> {tokens}")
    assert tokenizer.decode(tokens, itos) == sample


if __name__ == "__main__":
    main()
