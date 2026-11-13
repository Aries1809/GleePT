#!/usr/bin/env python3

from collections.abc import Iterable


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(tokens: Iterable[int], itos: dict[int, str]) -> str:
    return "".join(map(itos.__getitem__, tokens))
