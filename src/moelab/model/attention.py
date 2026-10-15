"""Causal grouped-query attention with rotary position embeddings."""

import torch
from torch import nn
from torch.nn import functional as F

from moelab.config import ModelConfig


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x):
        angles = torch.outer(torch.arange(x.size(2), device=x.device).float(), self.inv_freq)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        c = self.cfg
        q = self._rope(self.wq(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2))
        k = self._rope(self.wk(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2))
        v = self.wv(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        v = v.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=c.dropout if self.training else 0.0
        )
        return self.wo(y.transpose(1, 2).contiguous().view(b, t, c.d_model))
