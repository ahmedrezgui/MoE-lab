"""Serve the single demo model on localhost. Restart after training finishes."""

import os
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    checkpoint = Path("checkpoints/moe/last.pt")
    if not checkpoint.is_file():
        raise SystemExit("No model found. Run scripts/train.py first.")
    # This launcher intentionally selects the canonical model, ignoring stale shell overrides.
    os.environ["MOELAB_CHECKPOINT"] = str(checkpoint)
    os.environ.setdefault("MOELAB_DEVICE", "auto")
    uvicorn.run("moelab.serving.api:app", host="127.0.0.1", port=8000)
