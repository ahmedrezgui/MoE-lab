"""Stream document-separated JSONL into byte-token training artifacts."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from moelab.tokenizer.bpe import BPETokenizer


def documents(path):
    """Normalize line endings and outer whitespace; reject malformed records."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{number}: expected an object with string 'text'")
            text = record["text"].replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                raise ValueError(f"{path}:{number}: empty document")
            yield text


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_documents(train_source, val_source, output, block_size=128):
    """Keep validation intact; remove normalized exact duplicates from training.

    Documents are encoded separately and separated by two newline bytes. This is
    a text boundary, not EOS or an attention barrier: blocks may cross documents.
    Byte-only tokenization makes the vocabulary complete without fitting merges.
    """
    train_source, val_source, output = map(Path, (train_source, val_source, output))
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if train_source.resolve() == val_source.resolve():
        raise ValueError("Training and validation must use different source files")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    sources = {
        split: {"path": str(path), "sha256": fingerprint(path)}
        for split, path in (("train", train_source), ("val", val_source))
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    validation_hashes = set()
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        for split, source in (("val", val_source), ("train", train_source)):
            seen = set()
            counts = {"read": 0, "kept": 0, "duplicates": 0, "overlap_with_val": 0, "tokens": 0}
            with (staging / f"ids_{split}.bin").open("wb") as stream:
                for text in documents(source):
                    counts["read"] += 1
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in seen:
                        counts["duplicates"] += 1
                        continue
                    seen.add(digest)
                    if split == "train" and digest in validation_hashes:
                        counts["overlap_with_val"] += 1
                        continue
                    encoded = text.encode("utf-8") + b"\n\n"
                    np.frombuffer(encoded, dtype=np.uint8).astype("<u2").tofile(stream)
                    counts["kept"] += 1
                    counts["tokens"] += len(encoded)
            if counts["kept"] == 0 or counts["tokens"] <= block_size:
                raise ValueError(
                    f"{split} needs retained documents and at least block_size + 1 tokens"
                )
            if split == "val":
                validation_hashes = seen
            stats[split] = counts
        BPETokenizer(256).save(staging / "tokenizer.json")
        metadata = {
            "format_version": 1,
            "sources": sources,
            "split": "explicit document-level source files; validation takes precedence",
            "normalization": "CRLF/CR to LF; strip outer whitespace",
            "deduplication": "SHA-256 of normalized UTF-8; exact within/across splits",
            "boundary": "two newlines per document; not EOS; blocks may cross documents",
            "tokenizer": "UTF-8 bytes; no learned merges",
            "vocab_size": 256,
            "dtype": "little-endian uint16",
            "block_size_checked": block_size,
            "documents": stats,
            "tokens": {split: counts["tokens"] for split, counts in stats.items()},
            "artifacts_sha256": {
                name: fingerprint(staging / name)
                for name in ("ids_train.bin", "ids_val.bin", "tokenizer.json")
            },
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(output)
    return metadata
