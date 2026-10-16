"""Full-stream next-token scoring in consecutive, context-reset blocks."""

import math
import time

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate_full_stream(model, ids, batch_size=2):
    """Score positions 1..N-1 once; include the short tail and weight by targets."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if ids.ndim != 1 or len(ids) < 2:
        raise ValueError("Need a one-dimensional stream with at least two tokens")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Token IDs must be integers")
    if int(ids.min()) < 0 or int(ids.max()) >= model.cfg.vocab_size:
        raise ValueError("Token ID outside model vocabulary")
    parameter = next(model.parameters())
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Full evaluation requires fp32 model weights")
    device = parameter.device
    context = model.cfg.max_seq_len
    full_blocks, tail = divmod(len(ids) - 1, context)
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    total_nll, targets = 0.0, 0
    try:
        batches = []
        for first in range(0, full_blocks, batch_size):
            batches.append(
                [(i * context, context) for i in range(first, min(first + batch_size, full_blocks))]
            )
        if tail:
            batches.append([(full_blocks * context, tail)])
        for blocks in batches:
            x = np.stack([ids[offset : offset + size] for offset, size in blocks]).astype(np.int64)
            y = np.stack([ids[offset + 1 : offset + size + 1] for offset, size in blocks]).astype(
                np.int64
            )
            with torch.autocast(device.type, enabled=False):
                logits, _ = model(torch.from_numpy(x).to(device))
                loss = F.cross_entropy(
                    logits.float().flatten(0, 1),
                    torch.from_numpy(y).to(device).flatten(),
                    reduction="sum",
                )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite full-stream validation loss")
            total_nll += loss.item()
            targets += y.size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        model.train(was_training)
    seconds = time.perf_counter() - start
    ce = total_nll / targets
    return {
        "val_ce": ce,
        "val_perplexity": math.exp(ce),
        "summed_nll": total_nll,
        "stream_tokens": len(ids),
        "scored_targets": targets,
        "unscored_prefix_tokens": 1,
        "context_length": context,
        "full_blocks": full_blocks,
        "tail_targets": tail,
        "batch_size": batch_size,
        "evaluation_seconds": seconds,
        "targets_per_second": targets / seconds,
        "precision": "fp32",
        "includes_aux_loss": False,
        "policy": "consecutive context-reset blocks; target positions 1..N-1 scored once; "
        "short tail included; no padding; EOS and cross-document transitions included",
        "timing_scope": "forward scoring and transfers; synchronized CUDA boundaries; "
        "excludes checkpoint loading, file hashing and input validation",
    }
