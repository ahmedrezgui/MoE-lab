"""Readable byte-level BPE; intended for small educational corpora."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def merge(ids: list[int], pair: tuple[int, int], token: int) -> list[int]:
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(token)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class BPETokenizer:
    def __init__(self, vocab_size: int = 256) -> None:
        if not 256 <= vocab_size <= 65536:
            raise ValueError("vocab_size must fit uint16 and include all 256 bytes")
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}

    def train(self, text: str) -> None:
        self.merges = {}
        ids = list(text.encode("utf-8"))
        for token in range(256, self.vocab_size):
            counts = Counter(zip(ids, ids[1:], strict=False))
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < 2:
                break
            self.merges[pair] = token
            ids = merge(ids, pair, token)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for pair, token in self.merges.items():
            ids = merge(ids, pair, token)
        return ids

    def decode(self, ids: list[int]) -> str:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), token in self.merges.items():
            vocab[token] = vocab[a] + vocab[b]
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "vocab_size": self.vocab_size,
                    "merges": [[a, b, t] for (a, b), t in self.merges.items()],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(data["vocab_size"])
        for a, b, token in data["merges"]:
            if token != 256 + len(obj.merges) or not 0 <= a < token or not 0 <= b < token:
                raise ValueError("Invalid BPE merge table")
            if (a, b) in obj.merges or token >= obj.vocab_size:
                raise ValueError("Invalid BPE merge table")
            obj.merges[a, b] = token
        return obj
