"""Dropless sparse top-k routing, with a Switch-style balancing objective."""

import torch
from torch import nn
from torch.nn import functional as F

from moelab.config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, hidden_mult: float = 8 / 3, *, hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        hidden = int(d_model * hidden_mult) if hidden_dim is None else hidden_dim
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.moe is not None
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.moe.n_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(cfg.d_model, hidden_dim=cfg.effective_ffn_hidden_dim)
            for _ in range(cfg.moe.n_experts)
        )
        self.register_buffer("token_counts", torch.zeros(cfg.moe.n_experts), persistent=False)

    def forward(self, x):
        c = self.cfg.moe
        flat = x.reshape(-1, x.size(-1))
        probs = F.softmax(self.router(flat).float(), dim=-1)
        weights, selected = probs.topk(c.top_k, dim=-1)
        # For k=1 retain the probability so the task loss can train the router.
        if c.top_k > 1:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(selected == i)
            if rows.numel():
                values = expert(flat[rows]) * weights[rows, slots, None].to(flat.dtype)
                out.index_add_(0, rows, values)
        counts = torch.bincount(selected.flatten(), minlength=c.n_experts).float()
        self.token_counts.copy_(counts.detach())
        # Assignment fractions sum to one even for k>1; uniform routing has loss 1.
        fractions = counts / selected.numel()
        aux = c.n_experts * (fractions * probs.mean(dim=0)).sum()
        return out.view_as(x), aux
