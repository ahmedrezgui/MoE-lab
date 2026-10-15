"""Mathematical invariants and learning tests for the educational model."""

import pytest

torch = pytest.importorskip("torch")
from moelab.config import ModelConfig  # noqa: E402
from moelab.model.attention import Attention  # noqa: E402
from moelab.model.moe import MoELayer  # noqa: E402
from moelab.model.transformer import Transformer  # noqa: E402

TINY = dict(vocab_size=256, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32)


def cfg(moe=True, **kwargs):
    return ModelConfig(**(TINY | kwargs), moe={"n_experts": 4, "top_k": 2} if moe else None)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(123)


def test_attention_shape_and_causality():
    attn = Attention(cfg()).eval()
    x = torch.randn(2, 16, 64)
    before = attn(x)
    x[:, 8:] = torch.randn_like(x[:, 8:])
    after = attn(x)
    assert before.shape == x.shape
    torch.testing.assert_close(before[:, :8], after[:, :8])


def test_sparse_router_matches_dense_reference_and_has_gradients():
    layer = MoELayer(cfg())
    x = torch.randn(2, 7, 64, requires_grad=True)
    output, aux = layer(x)
    flat = x.reshape(-1, 64)
    probs = layer.router(flat).softmax(-1)
    weights, indices = probs.topk(2, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    dense_gates = torch.zeros_like(probs).scatter(1, indices, weights)
    expected = sum(
        expert(flat) * dense_gates[:, i : i + 1] for i, expert in enumerate(layer.experts)
    ).view_as(x)
    torch.testing.assert_close(output, expected)
    assert aux.ndim == 0 and aux > 0
    assert layer.token_counts.sum() == 2 * 7 * 2
    (output.square().mean() + aux * 0.01).backward()
    assert layer.router.weight.grad.abs().sum() > 0
    assert x.grad.isfinite().all()
    for expert in layer.experts:
        assert expert.w1.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("moe", [False, True])
def test_model_causality_and_parameter_counts(moe):
    model = Transformer(cfg(moe)).eval()
    x = torch.randint(0, 256, (2, 12))
    before, _ = model(x)
    x[:, 6:] = torch.randint(0, 256, (2, 6))
    after, _ = model(x)
    torch.testing.assert_close(before[:, :6], after[:, :6])
    assert model.num_params(True) <= model.num_params()
    if moe:
        expert_size = sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())
        assert model.num_params() - model.num_params(True) == 2 * expert_size


@pytest.mark.parametrize("moe", [False, True])
def test_overfit_one_batch(moe):
    model = Transformer(cfg(moe))
    x = torch.arange(16).reshape(1, -1)
    y = x + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = model(x, y)[1].item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.2
    assert loss.item() < initial * 0.1


def test_top_one_router_receives_task_gradient():
    config = ModelConfig(**TINY, moe={"n_experts": 4, "top_k": 1})
    layer = MoELayer(config)
    output, _ = layer(torch.randn(2, 8, 64))
    output.square().mean().backward()
    assert layer.router.weight.grad.abs().sum() > 0
