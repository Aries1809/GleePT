#!/usr/bin/env python3

def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(tokens: list[int], itos: dict[int, str]) -> str:
    return "".join([itos[t] for t in tokens])
