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
    args = parser.parse_args()
    result = prepare_documents(args.train_input, args.val_input, args.output, args.block_size)
    print(json.dumps(result, indent=2))
