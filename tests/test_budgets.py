from pathlib import Path

import pytest
from pydantic import ValidationError

from moelab.config import ModelConfig, load_model_config
from moelab.model.budget import compare_budgets, parameter_budget

CONFIGS = Path(__file__).resolve().parents[1] / "configs/model"


@pytest.mark.parametrize("name", ["compare-dense", "compare-moe", "laptop-moe"])
def test_budget_matches_allocated_parameters_and_backward(name):
    torch = pytest.importorskip("torch")
    from moelab.model.transformer import Transformer

    torch.set_num_threads(2)
    cfg = load_model_config(CONFIGS / f"{name}.yaml")
    model = Transformer(cfg)
    budget = parameter_budget(cfg)
    assert budget["total_params"] == sum(p.numel() for p in model.parameters())
    assert budget["active_params_estimate"] == model.num_params(active_only=True)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    loss.backward()
    assert model.embedding.weight.grad.isfinite().all()
    ffn = model.blocks[0].ffn
    expert = ffn.experts[0] if cfg.moe else ffn
    assert expert.w1.out_features == cfg.effective_ffn_hidden_dim


def test_pair_is_matched_with_router_and_tied_embedding_counted_once():
    dense = load_model_config(CONFIGS / "compare-dense.yaml")
    moe = load_model_config(CONFIGS / "compare-moe.yaml")
    report = compare_budgets(dense, moe)
    assert report["dense"]["total_params"] == 656256
    assert report["moe"]["total_params"] == 1180288
    assert report["moe"]["active_params_estimate"] == 656512
    assert report["active_gap_params"] == 256
    assert report["within_0_1_percent"]
    # Changing the common tokenizer changes both totals equally.
    for cfg in (dense, moe):
        cfg.vocab_size = 272
    changed = compare_budgets(dense, moe)
    assert changed["active_gap_params"] == 256
    assert changed["dense"]["embedding_params"] == 272 * 128
    dense.ffn_hidden_dim = 341
    assert not compare_budgets(dense, moe)["within_0_1_percent"]
    with pytest.raises(ValueError, match="dense model"):
        compare_budgets(moe, dense)
    dense.vocab_size = 256
    with pytest.raises(ValueError, match="vocab_size"):
        compare_budgets(dense, moe)


def test_default_width_preserves_old_architecture_and_rejects_zero():
    cfg = load_model_config(CONFIGS / "laptop-moe.yaml")
    assert cfg.ffn_hidden_dim is None and cfg.effective_ffn_hidden_dim == 341
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(cfg.model_dump() | {"ffn_hidden_dim": 0})
