"""Prepare local UTF-8 text; train BPE only on the training split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from moelab.tokenizer.bpe import BPETokenizer


def prepare(source, output, vocab_size=256, val_fraction=0.1):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    text = Path(source).read_text(encoding="utf-8")
    cut = int(len(text) * (1 - val_fraction))
    train_text, val_text = text[:cut], text[cut:]
    tokenizer = BPETokenizer(vocab_size)
    tokenizer.train(train_text)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    counts = {}
    for name, split in [("train", train_text), ("val", val_text)]:
        ids = tokenizer.encode(split)
        np.array(ids, dtype="<u2").tofile(output / f"ids_{name}.bin")
        counts[name] = len(ids)
    metadata = {
        "source": str(source),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "split": "contiguous character split before tokenizer training",
        "val_fraction": val_fraction,
        "vocab_size": vocab_size,
        "dtype": "little-endian uint16",
        "tokens": counts,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    prepare(args.input, args.output, args.vocab_size, args.val_fraction)
