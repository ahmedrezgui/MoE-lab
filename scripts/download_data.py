"""Download a bounded, revision-pinned subset of the original TinyStories splits."""

import argparse
import json

from moelab.data.download import REVISION, download_tinystories

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/tinystories")
    parser.add_argument("--train-documents", type=int, default=1000)
    parser.add_argument("--val-documents", type=int, default=100)
    parser.add_argument("--max-bytes-per-split", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-document-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    manifest = download_tinystories(
        args.output,
        args.train_documents,
        args.val_documents,
        args.max_bytes_per_split,
        args.max_document_bytes,
        args.revision,
    )
    print(json.dumps(manifest, indent=2))
