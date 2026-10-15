"""Config invariants — green from day one, and the pattern for every future test."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from moelab.config import ModelConfig, load_model_config, load_training_config

CONFIGS = Path(__file__).parent.parent / "configs"


def test_tiny_config_loads():
    cfg = load_model_config(CONFIGS / "model" / "tiny.yaml")
    assert cfg.head_dim == cfg.d_model // cfg.n_heads
    assert cfg.moe is None


def test_tiny_moe_config_loads():
    cfg = load_model_config(CONFIGS / "model" / "tiny-moe.yaml")
    assert cfg.moe is not None
    assert cfg.moe.top_k <= cfg.moe.n_experts


def test_training_config_loads():
    cfg = load_training_config(CONFIGS / "training" / "base.yaml")
    assert cfg.precision in {"bf16", "fp16", "fp32"}


def test_rejects_indivisible_heads():
    with pytest.raises(ValidationError):
        ModelConfig(
            vocab_size=8192, d_model=250, n_layers=2, n_heads=8, n_kv_heads=4, max_seq_len=128
        )


def test_rejects_bad_gqa_grouping():
    with pytest.raises(ValidationError):
        ModelConfig(
            vocab_size=8192, d_model=256, n_layers=2, n_heads=8, n_kv_heads=3, max_seq_len=128
        )


def test_rejects_topk_gt_experts():
    with pytest.raises(ValidationError):
        ModelConfig(
            vocab_size=8192,
            d_model=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=4,
            max_seq_len=128,
            moe={"n_experts": 4, "top_k": 8},
        )


def test_rejects_odd_rotary_dimension():
    with pytest.raises(ValidationError):
        ModelConfig(vocab_size=256, d_model=66, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=32)


def test_rejects_config_typo():
    with pytest.raises(ValidationError):
        ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            n_expert=4,
        )
