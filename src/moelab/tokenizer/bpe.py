"""Readable byte-level BPE with bounded, document-separated fitting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
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
    def __init__(self, vocab_size: int = 256, *, add_eos: bool = False) -> None:
        if not isinstance(add_eos, bool):
            raise ValueError("add_eos must be a boolean")
        if not 256 + int(add_eos) <= vocab_size <= 65536:
            raise ValueError("vocab_size must fit uint16 and include 256 bytes plus optional EOS")
        self.add_eos = add_eos
        self.requested_vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        """Number of decodable IDs, not the requested upper bound."""
        return 256 + len(self.merges) + int(self.add_eos)

    @property
    def eos_token_id(self) -> int | None:
        """EOS follows learned merge IDs; it never participates in BPE fitting."""
        return 256 + len(self.merges) if self.add_eos else None

    def train(self, text: str, max_training_bytes: int = 65536) -> dict:
        return self.train_documents([text], max_training_bytes)

    def train_documents(self, texts: Iterable[str], max_training_bytes: int = 65536) -> dict:
        """Fit on a bounded prefix; adjacent pairs never cross document boundaries.

        The final sampled document may be truncated at a byte boundary. This is
        valid for byte BPE; full documents are still encoded during preparation.
        The bound limits sample bytes, not total Python memory or elapsed time.
        """
        if max_training_bytes < 1:
            raise ValueError("max_training_bytes must be positive")
        self.merges = {}
        samples = []
        used = 0
        truncated = False
        digest = hashlib.sha256()
        merge_limit = self.requested_vocab_size - int(self.add_eos)
        if merge_limit > 256:
            for text in texts:
                raw = text.encode("utf-8")
                sample = raw[: max_training_bytes - used]
                if not sample:
                    continue
                samples.append(list(sample))
                used += len(sample)
                truncated = len(sample) < len(raw)
                # Length prefixes distinguish document boundaries in the hash.
                digest.update(len(sample).to_bytes(8, "big"))
                digest.update(sample)
                if used == max_training_bytes:
                    break
        sample_count = len(samples)
        for token in range(256, merge_limit):
            counts = Counter()
            for ids in samples:
                counts.update(zip(ids, ids[1:], strict=False))
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < 2:
                break
            self.merges[pair] = token
            samples = [merge(ids, pair, token) for ids in samples]
        return {
            "bytes": used,
            "documents": sample_count,
            "last_document_truncated": truncated,
            "sha256": digest.hexdigest(),
        }

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        if add_eos and self.eos_token_id is None:
            raise ValueError("Tokenizer has no EOS token")
        ids = list(text.encode("utf-8"))
        for pair, token in self.merges.items():
            ids = merge(ids, pair, token)
        return ids + [self.eos_token_id] if add_eos else ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), token in self.merges.items():
            vocab[token] = vocab[a] + vocab[b]
        if self.eos_token_id is not None:
            vocab[self.eos_token_id] = b"" if skip_special_tokens else b"<|endoftext|>"
        if any(i not in vocab for i in ids):
            raise ValueError(
                f"Token IDs must be within the actual vocabulary [0, {self.vocab_size})"
            )
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "format_version": 3,
                    "add_eos": self.add_eos,
                    "eos_token_id": self.eos_token_id,
                    "requested_vocab_size": self.requested_vocab_size,
                    "vocab_size": self.vocab_size,
                    "merges": [[a, b, t] for (a, b), t in self.merges.items()],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("format_version", 1)
        if version not in (1, 2, 3):
            raise ValueError("Unsupported tokenizer format version")
        # Legacy files stored the requested capacity under vocab_size.
        obj = cls(
            data["requested_vocab_size"] if version >= 2 else data["vocab_size"],
            add_eos=data["add_eos"] if version == 3 else False,
        )
        for a, b, token in data["merges"]:
            if token != 256 + len(obj.merges) or not 0 <= a < token or not 0 <= b < token:
                raise ValueError("Invalid BPE merge table")
            if (a, b) in obj.merges or token >= obj.requested_vocab_size - int(obj.add_eos):
                raise ValueError("Invalid BPE merge table")
            obj.merges[a, b] = token
        if version >= 2 and data["vocab_size"] != obj.vocab_size:
            raise ValueError("Stored vocabulary size does not match the merge table")
        if version == 3 and data["eos_token_id"] != obj.eos_token_id:
            raise ValueError("Stored EOS ID does not match the merge table")
        return obj
