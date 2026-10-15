"""Download and prepare the raw corpus (Phase 1).

Start: TinyStories (HF `roneneldan/TinyStories`). Later: swap in your own
domain corpus — a French/English financial-text mix ties this repo to your
BNP story.

Pipeline (make each step a DVC stage in dvc.yaml):
  raw text -> train tokenizer -> encode -> data/ids_{train,val}.bin
"""

from __future__ import annotations

if __name__ == "__main__":
    raise NotImplementedError("Phase 1")
