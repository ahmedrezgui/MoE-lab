"""Random next-token blocks backed by a uint16 token file."""

from pathlib import Path

import numpy as np


class BlockDataset:
    def __init__(self, ids_path: str | Path, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if Path(ids_path).stat().st_size < 2 * (block_size + 1):
            raise ValueError("Corpus needs at least block_size + 1 tokens")
        self.ids = np.memmap(ids_path, dtype="<u2", mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def get_batch(self, batch_size: int, rng: np.random.Generator):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        starts = rng.integers(0, len(self), size=batch_size)
        blocks = np.stack([self.ids[i : i + self.block_size + 1] for i in starts]).astype(np.int64)
        return blocks[:, :-1].copy(), blocks[:, 1:].copy()
