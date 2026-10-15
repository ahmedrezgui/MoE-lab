"""Pre-norm decoder with tied embeddings and dense or sparse SwiGLU blocks."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from moelab.config import ModelConfig
from moelab.model.attention import Attention
from moelab.model.moe import MoELayer, SwiGLU


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = MoELayer(cfg) if cfg.moe else SwiGLU(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        h = self.ffn_norm(x)
        if isinstance(self.ffn, MoELayer):
            h, aux = self.ffn(h)
        else:
            h, aux = self.ffn(h), x.new_zeros(())
        return x + h, aux


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)
        for name, param in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 1 <= idx.size(1) <= self.cfg.max_seq_len:
            raise ValueError("Expected [batch, sequence] within configured context length")
        x = self.embedding(idx)
        aux = x.new_zeros(())
        for block in self.blocks:
            x, block_aux = block(x)
            aux = aux + block_aux
        logits = F.linear(self.norm(x), self.embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.reshape(-1))
            if self.cfg.moe:
                loss = loss + self.cfg.moe.load_balance_coef * aux / self.cfg.n_layers
        return logits, loss

    def num_params(self, active_only: bool = False) -> int:
        """Unique weights; active estimate includes shared weights and k experts/layer."""
        total = sum(p.numel() for p in self.parameters())
        if active_only and self.cfg.moe:
            c = self.cfg.moe
            for block in self.blocks:
                expert_size = sum(p.numel() for p in block.ffn.experts[0].parameters())
                total -= (c.n_experts - c.top_k) * expert_size
        return total
