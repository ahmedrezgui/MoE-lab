"""Typed, validated configs for every stage of the pipeline.

This module is deliberately torch-free so config validation stays fast and
importable everywhere (CI, DVC stages, serving). All invariants that would
otherwise blow up mid-training are rejected here, at load time.

Interview note: "model-level invariants enforced at validation time" is the
same pattern you used in the BNP advisory system — point that out.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MoEConfig(StrictConfig):
    """Mixture-of-Experts settings for the FFN blocks."""

    n_experts: int = Field(ge=2)
    top_k: int = Field(ge=1)
    load_balance_coef: float = Field(ge=0.0, default=0.01)

    @model_validator(mode="after")
    def _check_topk(self) -> MoEConfig:
        if self.top_k > self.n_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed n_experts ({self.n_experts})")
        return self


class ModelConfig(StrictConfig):
    """Architecture of the decoder-only transformer."""

    vocab_size: int = Field(ge=256)
    d_model: int = Field(ge=64)
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    n_kv_heads: int = Field(ge=1)
    max_seq_len: int = Field(ge=32)
    dropout: float = Field(ge=0.0, le=0.5, default=0.0)
    moe: MoEConfig | None = None

    @model_validator(mode="after")
    def _check_dims(self) -> ModelConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads}) for GQA"
            )
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        return self

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class TrainingConfig(StrictConfig):
    """Optimizer, schedule, and run bookkeeping."""

    seed: int = 42
    lr: float = Field(gt=0.0)
    warmup_steps: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    grad_accum_steps: int = Field(ge=1, default=1)
    weight_decay: float = Field(ge=0.0, default=0.0)
    grad_clip: float | None = Field(default=1.0, gt=0.0)
    precision: str = Field(default="bf16", pattern="^(bf16|fp16|fp32)$")
    eval_every: int = Field(ge=1, default=250)
    checkpoint_every: int = Field(ge=1, default=1000)
    mlflow_experiment: str = "moelab"
    run_name: str | None = None


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(load_yaml(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.model_validate(load_yaml(path))
