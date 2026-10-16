"""Prepare document-separated JSONL with optional bounded byte-level BPE."""

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


def prepare_documents(
    train_source,
    val_source,
    output,
    block_size=128,
    *,
    vocab_size=256,
    tokenizer_training_bytes=65536,
    add_eos=False,
):
    """Keep validation intact; remove normalized exact duplicates from training.

    Documents are encoded separately and followed by EOS when enabled, otherwise
    two newline bytes. Neither creates an attention barrier: blocks may cross documents.
    Optional BPE fits only a bounded prefix of retained training documents.
    """
    tokenizer = BPETokenizer(vocab_size, add_eos=add_eos)
    if tokenizer_training_bytes < 1:
        raise ValueError("tokenizer_training_bytes must be positive")
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
            with (staging / f"kept_{split}.jsonl").open("w", encoding="utf-8") as stream:
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
                    stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    counts["kept"] += 1
            if counts["kept"] == 0:
                raise ValueError(f"{split} needs retained documents")
            if split == "val":
                validation_hashes = seen
            stats[split] = counts
        sample = tokenizer.train_documents(
            documents(staging / "kept_train.jsonl"), tokenizer_training_bytes
        )
        boundary_ids = [tokenizer.eos_token_id] if add_eos else tokenizer.encode("\n\n")
        for split in ("val", "train"):
            spool = staging / f"kept_{split}.jsonl"
            with (staging / f"ids_{split}.bin").open("wb") as stream:
                for text in documents(spool):
                    ids = tokenizer.encode(text) + boundary_ids
                    np.asarray(ids, dtype="<u2").tofile(stream)
                    stats[split]["tokens"] += len(ids)
            if stats[split]["tokens"] <= block_size:
                raise ValueError(f"{split} needs at least block_size + 1 tokens after tokenization")
            spool.unlink()
        tokenizer.save(staging / "tokenizer.json")
        metadata = {
            "format_version": 3,
            "sources": sources,
            "split": "explicit document-level source files; validation takes precedence",
            "normalization": "CRLF/CR to LF; strip outer whitespace",
            "deduplication": "SHA-256 of normalized UTF-8; exact within/across splits",
            "boundary": (
                "one EOS after each document; no attention barrier"
                if add_eos
                else "two newlines encoded separately per document; no attention barrier"
            ),
            "eos_token_id": tokenizer.eos_token_id,
            "tokenizer": "byte-level BPE" if tokenizer.merges else "UTF-8 bytes; no learned merges",
            "requested_vocab_size": vocab_size,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_training": {
                "max_bytes": tokenizer_training_bytes,
                **sample,
                "selection": "prefix of deduplicated, overlap-free training documents only",
                "boundary_policy": "no pair counts or merges across document boundaries",
                "hash_scope": "sample documents, each prefixed by an 8-byte big-endian byte length",
            },
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
