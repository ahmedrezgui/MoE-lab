import json

import numpy as np
import pytest

from moelab.data.dataset import BlockDataset
from moelab.data.prepare import prepare_documents
from moelab.tokenizer.bpe import BPETokenizer


def write_jsonl(path, texts):
    path.write_text("\n".join(json.dumps({"text": text}) for text in texts), encoding="utf-8")
    return path


def test_document_splits_deduplicate_and_preserve_validation(tmp_path):
    train = write_jsonl(
        tmp_path / "train.jsonl",
        ["shared story", "  train\r\nstory  ", "train\nstory", "new story"],
    )
    val = write_jsonl(tmp_path / "val.jsonl", [" shared story ", "shared story", "held-out"])
    output = tmp_path / "prepared"
    metadata = prepare_documents(train, val, output, block_size=4)
    tokenizer = BPETokenizer.load(output / "tokenizer.json")

    def decode(split):
        return tokenizer.decode(np.fromfile(output / f"ids_{split}.bin", dtype="<u2").tolist())

    assert decode("train") == "train\nstory\n\nnew story\n\n"
    assert decode("val") == "shared story\n\nheld-out\n\n"
    assert metadata["documents"]["train"] == {
        "read": 4,
        "kept": 2,
        "duplicates": 1,
        "overlap_with_val": 1,
        "tokens": 24,
    }
    assert metadata["documents"]["val"]["duplicates"] == 1
    assert not tokenizer.merges
    assert len(BlockDataset(output / "ids_train.bin", 4)) == 20
    repeated = tmp_path / "repeated"
    second = prepare_documents(train, val, repeated, block_size=4)
    assert metadata == second
    for name in ("ids_train.bin", "ids_val.bin", "tokenizer.json", "metadata.json"):
        assert (output / name).read_bytes() == (repeated / name).read_bytes()


@pytest.mark.parametrize("record", ["{broken", '{"text": 42}', '{"text": "  "}'])
def test_bad_document_leaves_no_partial_output(tmp_path, record):
    train = tmp_path / "train.jsonl"
    train.write_text(record, encoding="utf-8")
    val = write_jsonl(tmp_path / "val.jsonl", ["validation story"])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="train.jsonl:1"):
        prepare_documents(train, val, output, block_size=4)
    assert not output.exists()
    assert not list(tmp_path.glob(".prepare-*"))


def test_rejects_overlap_only_train_and_existing_output(tmp_path):
    train = write_jsonl(tmp_path / "train.jsonl", ["same story"])
    val = write_jsonl(tmp_path / "val.jsonl", ["same story"])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="train needs"):
        prepare_documents(train, val, output, block_size=4)
    assert not output.exists()
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        prepare_documents(train, val, output)
    assert sentinel.read_text() == "keep"


def test_unicode_bytes_and_short_split(tmp_path):
    train = write_jsonl(tmp_path / "train.jsonl", ["caf\u00e9 \U0001f98a"])
    val = write_jsonl(tmp_path / "val.jsonl", ["other story"])
    output = tmp_path / "prepared"
    prepare_documents(train, val, output, block_size=4)
    tokenizer = BPETokenizer.load(output / "tokenizer.json")
    ids = np.fromfile(output / "ids_train.bin", dtype="<u2").tolist()
    assert tokenizer.decode(ids) == "caf\u00e9 \U0001f98a\n\n"
    with pytest.raises(ValueError, match="val needs"):
        prepare_documents(train, val, tmp_path / "too-short", block_size=100)
    with pytest.raises(ValueError, match="different source"):
        prepare_documents(train, train, tmp_path / "same-file", block_size=4)
