"""KV cache for autoregressive generation.

BUILD (Phase 3): preallocate (n_layers, 2, batch, n_kv_heads, max_seq_len,
head_dim), append per step, and pass position offsets to RoPE.

Benchmark to publish: tokens/sec with vs without cache at seq_len 256/512.
This single number is why inference is memory-bound — say that in the README.
"""

from __future__ import annotations


class KVCache:
    def __init__(self, n_layers: int, batch: int, n_kv_heads: int, max_seq_len: int, head_dim: int):
        raise NotImplementedError("Phase 3")
