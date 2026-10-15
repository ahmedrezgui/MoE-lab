"""Uncached autoregressive reference sampler with a sliding context window."""

import torch


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("max_new_tokens and temperature must be nonnegative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if idx.ndim != 2 or idx.size(1) == 0:
        raise ValueError("Provide a nonempty [batch, sequence] prompt")
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.cfg.max_seq_len :])
            scores = logits[:, -1].float()
            if temperature == 0:
                token = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k is not None:
                    threshold = scores.topk(min(top_k, scores.size(-1))).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -torch.inf)
                if top_p is not None:
                    sorted_scores, order = scores.sort(descending=True)
                    remove = sorted_scores.softmax(-1).cumsum(-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    scores = scores.scatter(1, order, sorted_scores.masked_fill(remove, -torch.inf))
                token = torch.multinomial(scores.softmax(-1), num_samples=1)
            idx = torch.cat((idx, token), dim=1)
    finally:
        model.train(was_training)
    return idx
