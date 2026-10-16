"""Reserve a fixed, overlap-filtered confirmation slice with a frozen tokenizer."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from moelab.data.prepare import documents, fingerprint
from moelab.tokenizer.bpe import BPETokenizer


def prepare_confirmation(
    source,
    prior_train,
    prior_val,
    training_data_dir,
    output,
    start_document=100,
    document_count=200,
):
    source, prior_train, prior_val, training, output = map(
        Path, (source, prior_train, prior_val, training_data_dir, output)
    )
    if start_document < 1 or document_count < 1:
        raise ValueError("Positive slice start and document count required")
    if output.exists():
        raise ValueError("Output already exists")
    original = json.loads((training / "metadata.json").read_text(encoding="utf-8"))
    if (
        fingerprint(prior_train) != original["sources"]["train"]["sha256"]
        or fingerprint(prior_val) != original["sources"]["val"]["sha256"]
    ):
        raise ValueError("Prior raw splits do not match the training preparation metadata")
    hashes = {
        name: fingerprint(training / name)
        for name in ("ids_train.bin", "ids_val.bin", "tokenizer.json")
    }
    if hashes != original["artifacts_sha256"]:
        raise ValueError("Original training artifacts changed")
    train_texts = list(documents(prior_train))
    dev_texts = list(documents(prior_val))
    candidates = list(documents(source))
    if start_document != len(dev_texts) or candidates[:start_document] != dev_texts:
        raise ValueError("Source prefix must exactly match all prior development documents")
    selected = candidates[start_document : start_document + document_count]
    if len(selected) != document_count:
        raise ValueError("Insufficient documents for the fixed confirmation slice")

    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    train_hashes = {digest(text) for text in train_texts}
    dev_hashes = {digest(text) for text in dev_texts}
    seen = set()
    counts = {
        "requested": document_count,
        "kept": 0,
        "train_overlap": 0,
        "development_overlap": 0,
        "duplicates_within_slice": 0,
    }
    retained = []
    for text in selected:
        key = digest(text)
        if key in train_hashes:
            counts["train_overlap"] += 1
        elif key in dev_hashes:
            counts["development_overlap"] += 1
        elif key in seen:
            counts["duplicates_within_slice"] += 1
        else:
            retained.append(text)
            seen.add(key)
    counts["kept"] = len(retained)
    if not retained:
        raise ValueError("No disjoint documents remain")
    tokenizer = BPETokenizer.load(training / "tokenizer.json")
    boundary = (
        [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else tokenizer.encode("\n\n")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".confirmation-", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        token_count = 0
        with (staging / "ids_confirmation.bin").open("wb") as stream:
            for text in retained:
                ids = tokenizer.encode(text) + boundary
                np.asarray(ids, dtype="<u2").tofile(stream)
                token_count += len(ids)
        (staging / "tokenizer.json").write_bytes((training / "tokenizer.json").read_bytes())
        metadata = {
            "format_version": 1,
            "selection": {
                "start_document_zero_based": start_document,
                "candidate_documents": document_count,
                "replacement_policy": "drop overlaps; no refill",
            },
            "counts": counts,
            "tokens": token_count,
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id,
            "normalization": "CRLF/CR to LF; strip outer whitespace",
            "overlap_check": "SHA-256 normalized exact text; train precedence, "
            "then dev, then duplicates",
            "limitation": "No semantic/near-duplicate detection; "
            "fixed prefix is not a random sample",
            "source_sha256": fingerprint(source),
            "prior_raw_sha256": {"train": fingerprint(prior_train), "val": fingerprint(prior_val)},
            "training_data_hashes": hashes,
            "tokenizer_policy": "byte-for-byte frozen; no fitting",
            "retained_document_sha256": sorted(seen),
            "artifacts_sha256": {
                name: fingerprint(staging / name)
                for name in ("ids_confirmation.bin", "tokenizer.json")
            },
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(output)
    return metadata
