#!/usr/bin/env python3

import numpy as np
from numpy.typing import NDArray
import tokenizer as tkn
from setup import DATA_PATH, build_vocab, load_text


def encode_to_array(text: str, stoi: dict[str, int]) -> NDArray[np.int64]:
    return np.fromiter(map(stoi.__getitem__, text), dtype=np.int64, count=len(text))

def split_array(data: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    index: int = int(len(data) * 0.9)
    return np.split(data, [index])

def count_bigrams(data: NDArray[np.int64], vocab_size: int) -> NDArray[np.int64]:
    pairs = data[:-1] * vocab_size + data[1:]
    return np.bincount(pairs, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size)

def normalize(N: NDArray[np.int64], smooth: float = 1) -> NDArray[np.float64]:
    P = (N + smooth).astype(np.float64)
    P /= P.sum(axis=1, keepdims=True)
    return P

if __name__ == "__main__":
    text: str = load_text(DATA_PATH)
    stoi, itos = build_vocab(text)
    data: NDArray[np.int64] = encode_to_array(text, stoi)
    train, val = split_array(data)
    vocab_size: int = len(itos)
    N = count_bigrams(train, vocab_size)
    P = normalize(N)


    print(data.shape)
    print(data.dtype)
    print(data[:20])
    print(tkn.decode(data[:20], itos))
    print(f"train: {len(train)}\nval: {len(val)}")
    assert len(train) + len(val) == len(data)
    print(N.shape)
    print(N.dtype)
    print(N.sum())
    assert N.sum() == len(train) - 1
    print(N[stoi['e']])
    
    top = np.argsort(N, axis=None)[-10:][::-1]
    print(top)
    for i in top:
        x, y = divmod(i, vocab_size)
        print(f"{N[x, y]}: {itos[x]!r} -> {itos[y]!r}")

    print(N.sum(axis=0, keepdims=True))
    print(N.sum(axis=1, keepdims=True))
    assert np.allclose(P.sum(axis=1), 1)
    print(P[stoi['q'], stoi['u']])
