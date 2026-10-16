"""Bounded, revision-pinned TinyStories downloads using the standard library."""

import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

from moelab.data.prepare import fingerprint

DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
LICENSE = "cdla-sharing-1.0"
LICENSE_URL = "https://cdla.dev/sharing-1-0/"
DELIMITER = b"<|endoftext|>"


def open_source(url, max_bytes):
    request = Request(
        url,
        headers={
            "Range": f"bytes=0-{max_bytes - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "moelab/0.1",
        },
    )
    return urlopen(request, timeout=30)


def export_split(url, destination, count, max_bytes, max_document_bytes):
    """Write only complete delimited stories; fail rather than publish a short split."""
    consumed = 0
    written = 0
    buffer = b""
    digest = hashlib.sha256()
    with open_source(url, max_bytes) as source, destination.open("wb") as output:
        while consumed < max_bytes:
            chunk = source.read(min(65536, max_bytes - consumed))
            if not chunk:
                break
            consumed += len(chunk)
            digest.update(chunk)
            buffer += chunk
            while DELIMITER in buffer:
                story, buffer = buffer.split(DELIMITER, 1)
                if len(story) > max_document_bytes:
                    raise ValueError("Source document exceeds max_document_bytes")
                text = story.decode("utf-8").strip()
                if not text:
                    continue
                output.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode())
                written += 1
                if written == count:
                    return {
                        "documents": written,
                        "source_bytes_read": consumed,
                        "source_prefix_sha256": digest.hexdigest(),
                    }
            # Allow a delimiter split across reads, but bound the pending story.
            if len(buffer) > max_document_bytes + len(DELIMITER) - 1:
                raise ValueError("Source document exceeds max_document_bytes")
    raise ValueError(f"Only {written}/{count} complete documents within source EOF/byte limit")


def download_tinystories(
    output,
    train_documents=1000,
    val_documents=100,
    max_bytes_per_split=32 * 1024 * 1024,
    max_document_bytes=1024 * 1024,
    revision=REVISION,
):
    """Publish JSONL plus provenance only after both bounded downloads succeed."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an immutable 40-character lowercase commit SHA")
    if min(train_documents, val_documents, max_bytes_per_split, max_document_bytes) < 1:
        raise ValueError("Document counts and byte limits must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}"
    card_url = f"{base}/README.md"
    # Metadata is separately bounded; the extra byte detects an oversized card.
    with open_source(card_url, 65537) as source:
        card = source.read(65537)
    if len(card) > 65536:
        raise ValueError("Dataset card exceeds 64 KiB")
    card_text = card.decode("utf-8")
    parts = card_text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("Dataset card is missing YAML front matter")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict) or metadata.get("license") != LICENSE:
        raise ValueError("Dataset license differs from the reviewed CDLA-Sharing-1.0 source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        (staging / "SOURCE_README.md").write_bytes(card)
        splits = {}
        for split, filename, count in (
            ("train", "TinyStories-train.txt", train_documents),
            ("val", "TinyStories-valid.txt", val_documents),
        ):
            url = f"{base}/{filename}"
            destination = staging / f"{split}.jsonl"
            stats = export_split(url, destination, count, max_bytes_per_split, max_document_bytes)
            splits[split] = {
                "url": url,
                "source_file": filename,
                **stats,
                "output_file": destination.name,
                "sha256": fingerprint(destination),
            }
        manifest = {
            "format_version": 1,
            "dataset": DATASET,
            "revision": revision,
            "dataset_url": f"https://huggingface.co/datasets/{DATASET}",
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "dataset_card": {
                "url": card_url,
                "file": "SOURCE_README.md",
                "sha256": hashlib.sha256(card).hexdigest(),
            },
            "selection": "first N nonempty complete documents in each original split; not random",
            "modifications": "split on <|endoftext|>; strip outer whitespace; convert to JSONL",
            "limits": {
                "max_bytes_per_split": max_bytes_per_split,
                "max_document_bytes": max_document_bytes,
            },
            "source_hash_scope": "read prefix; may include bytes beyond the last selected story",
            "splits": splits,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        notice = (
            f"Source: {DATASET}\nRevision: {revision}\n"
            f"License: {LICENSE}\nLicense text: {LICENSE_URL}\n"
            "Subset converted from delimited UTF-8 text into JSONL; outer whitespace stripped.\n"
            "Original dataset card is retained as SOURCE_README.md.\n"
        )
        (staging / "NOTICE.txt").write_text(notice, encoding="utf-8")
        staging.rename(output)
    return manifest
