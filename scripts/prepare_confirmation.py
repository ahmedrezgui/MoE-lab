"""Prepare a fixed confirmation slice without refitting the tokenizer."""

import argparse
import json

from moelab.data.confirmation import prepare_confirmation

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--prior-train", required=True)
    p.add_argument("--prior-val", required=True)
    p.add_argument("--training-data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-document", type=int, default=100, help="Zero-based slice start")
    p.add_argument("--document-count", type=int, default=200)
    a = p.parse_args()
    print(
        json.dumps(
            prepare_confirmation(
                a.source,
                a.prior_train,
                a.prior_val,
                a.training_data_dir,
                a.output,
                a.start_document,
                a.document_count,
            ),
            indent=2,
        )
    )
