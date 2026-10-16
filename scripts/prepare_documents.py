"""Prepare separate JSONL document splits with exact deduplication."""

import argparse
import json

from moelab.data.prepare import prepare_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True, help="JSONL with a text field per record")
    parser.add_argument("--val-input", required=True, help="Separate held-out JSONL documents")
    parser.add_argument("--output", required=True, help="New artifact directory")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--vocab-size", type=int, default=256, help="Requested vocabulary upper bound"
    )
    parser.add_argument("--tokenizer-training-bytes", type=int, default=65536)
    parser.add_argument(
        "--add-eos", action="store_true", help="Reserve one vocabulary entry for EOS"
    )
    args = parser.parse_args()
    result = prepare_documents(
        args.train_input,
        args.val_input,
        args.output,
        args.block_size,
        vocab_size=args.vocab_size,
        tokenizer_training_bytes=args.tokenizer_training_bytes,
        add_eos=args.add_eos,
    )
    print(json.dumps(result, indent=2))
