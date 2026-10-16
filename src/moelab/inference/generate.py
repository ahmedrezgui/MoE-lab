"""Uncached autoregressive reference sampler with a sliding context window."""

import torch


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    *,
    eos_token_id=None,
    generator=None,
):
    """Include sampled EOS in results; pad finished batch rows with EOS.

    Only newly generated EOS stops a row. Prompt EOS tokens remain valid context.
    Without an EOS ID, preserve fixed-length generation behavior.
    """
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("max_new_tokens and temperature must be nonnegative")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if idx.ndim != 2 or idx.size(0) == 0 or idx.size(1) == 0:
        raise ValueError("Provide a nonempty [batch, sequence] prompt")
    if eos_token_id is not None and (
        type(eos_token_id) is not int or not 0 <= eos_token_id < model.cfg.vocab_size
    ):
        raise ValueError("eos_token_id must be an integer within the model vocabulary")
    finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
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
                token = torch.multinomial(scores.softmax(-1), num_samples=1, generator=generator)
            if eos_token_id is not None:
                token = token.masked_fill(finished[:, None], eos_token_id)
                finished |= token.squeeze(-1) == eos_token_id
            idx = torch.cat((idx, token), dim=1)
            if eos_token_id is not None and finished.all():
                break
    finally:
        model.train(was_training)
    return idx
