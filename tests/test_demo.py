import json

import pytest
import torch

from moelab.config import ModelConfig
from moelab.model.transformer import Transformer
from moelab.serving.demo import DemoModel
from moelab.tokenizer.bpe import BPETokenizer
from moelab.training.loop import file_hash


@pytest.fixture
def engine(tmp_path):
    torch.set_num_threads(2)
    cfg = ModelConfig(
        vocab_size=257, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32
    )
    tokenizer = tmp_path / "tokenizer.json"
    BPETokenizer(257, add_eos=True).save(tokenizer)
    path = tmp_path / "last.pt"
    torch.save(
        {
            "model_config": cfg.model_dump(),
            "model": Transformer(cfg).state_dict(),
            "step": 2,
            "training_config": {"seed": 42},
            "data_hashes": {"tokenizer.json": file_hash(tokenizer)},
        },
        path,
    )
    return DemoModel(path)


def test_demo_request_rng_is_repeatable_and_does_not_change_global_state(engine):
    before = torch.get_rng_state().clone()
    first = engine.complete("Hi", 8, 0.8, 40, 0.95, 42)
    engine.complete("other", 4, 0.8, 40, 0.95, 17)
    repeated = engine.complete("Hi", 8, 0.8, 40, 0.95, 42)
    assert first["text"] == repeated["text"]
    assert first["generated_tokens"] == repeated["generated_tokens"]
    assert torch.equal(before, torch.get_rng_state())
    assert first["text"].startswith("Hi")
    assert first["generated_tokens"] <= 8 and first["finish_reason"] in ("eos", "length")
    assert not engine.model.training


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": ""},
        {"prompt": " "},
        {"prompt": "a" * 33},
        {"max_new_tokens": 0},
        {"max_new_tokens": 257},
        {"temperature": float("nan")},
        {"top_k": -1},
        {"top_p": 0},
        {"seed": -1},
        {"seed": 1.5},
    ],
)
def test_demo_invalid_requests(engine, kwargs):
    with pytest.raises(ValueError):
        engine.complete(**({"prompt": "Hi"} | kwargs))


def test_demo_eos_count_and_hidden_special(engine):
    def forward(idx):
        scores = torch.zeros(*idx.shape, 257)
        scores[..., 256] = 100
        return scores, None

    engine.model.forward = forward
    result = engine.complete("Hi", 64, 0, 0, 1, 42)
    assert result["text"] == "Hi" and result["continuation"] == ""
    assert result["generated_tokens"] == 1 and result["finish_reason"] == "eos"


def test_tokenizer_with_same_size_but_different_bytes_is_rejected(engine):
    from pathlib import Path

    checkpoint = Path(engine.info["checkpoint"])
    tokenizer = checkpoint.with_name("tokenizer.json")
    payload = json.loads(tokenizer.read_text())
    payload["requested_vocab_size"] = 300
    tokenizer.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        DemoModel(checkpoint)


def test_gradio_builds_without_launching(engine):
    pytest.importorskip("gradio")
    from moelab.serving.ui import build_ui

    app = build_ui(engine)
    assert app.title == "MoE Lab | Story playground"
    assert any(
        dependency.get("api_name") == "generate" for dependency in app.config["dependencies"]
    )
    app.close()
