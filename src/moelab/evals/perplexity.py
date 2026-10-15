"""Held-out next-token cross entropy, excluding the router auxiliary loss."""

import math

import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(model, dataset, batches=10, batch_size=4, seed=123):
    if batches < 1 or batch_size < 1:
        raise ValueError("Evaluation needs positive batches and batch_size")
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    total = 0.0
    try:
        for _ in range(batches):
            x, y = dataset.get_batch(batch_size, rng)
            logits, _ = model(torch.from_numpy(x).to(device))
            total += F.cross_entropy(
                logits.float().flatten(0, 1), torch.from_numpy(y).to(device).reshape(-1)
            ).item()
    finally:
        model.train(was_training)
    ce = total / batches
    return {"val_ce": ce, "val_perplexity": math.exp(ce)}


def perplexity(model, dataset) -> float:
    return evaluate(model, dataset)["val_perplexity"]
