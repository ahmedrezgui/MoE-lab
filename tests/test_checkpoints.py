"""Checkpoint selection follows held-out CE, including across resumed runs."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from moelab.config import ModelConfig, TrainingConfig  # noqa: E402
from moelab.tokenizer.bpe import BPETokenizer  # noqa: E402
from moelab.training import loop  # noqa: E402


@pytest.fixture
def run_setup(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    BPETokenizer().save(data / "tokenizer.json")
    for split in ("train", "val"):
        np.tile(np.arange(40, dtype="<u2"), 4).tofile(data / f"ids_{split}.bin")
    model = ModelConfig(
        vocab_size=256, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2, max_seq_len=32
    )
    training = TrainingConfig(
        lr=0.003,
        warmup_steps=0,
        max_steps=6,
        batch_size=1,
        precision="fp32",
        eval_every=1,
        checkpoint_every=3,
    )
    return model, training, {"data_dir": data, "output_dir": tmp_path / "run"}


def test_best_checkpoint_survives_worsening_ties_and_resume(run_setup, monkeypatch):
    model_cfg, training_cfg, paths = run_setup
    # Step zero is diagnostic only, even when its score is lower than trained scores.
    scores = iter([0.5, 3.0, 2.0, 4.0, 2.0, 5.0, 1.0])
    evaluated_weights = []

    def evaluate(model, *args, **kwargs):
        evaluated_weights.append({k: v.clone() for k, v in model.state_dict().items()})
        return {"val_ce": next(scores), "val_perplexity": 10.0}

    monkeypatch.setattr(loop, "evaluate", evaluate)
    output = paths["output_dir"]
    loop.train(model_cfg, training_cfg, max_steps=3, **paths)
    best = torch.load(output / "best.pt", weights_only=True)
    last = torch.load(output / "last.pt", weights_only=True)
    assert (best["step"], best["best_step"], best["best_val_ce"]) == (2, 2, 2.0)
    assert (last["step"], last["best_step"]) == (3, 2)
    for name, weight in best["model"].items():
        torch.testing.assert_close(weight, evaluated_weights[2][name], rtol=0, atol=0)
    # The best evaluation happened between regular checkpoint intervals.
    loaded, _ = loop.load_checkpoint(output / "best.pt")
    assert loaded(torch.tensor([[1, 2]]))[0].shape == (1, 2, 256)
    original_best = (output / "best.pt").read_bytes()
    loop.train(model_cfg, training_cfg, max_steps=5, resume=output / "last.pt", **paths)
    assert (output / "best.pt").read_bytes() == original_best
    last = torch.load(output / "last.pt", weights_only=True)
    assert (last["step"], last["best_step"], last["best_val_ce"]) == (5, 2, 2.0)
    loop.train(model_cfg, training_cfg, resume=output / "last.pt", **paths)
    best = torch.load(output / "best.pt", weights_only=True)
    assert (best["step"], best["best_step"], best["best_val_ce"]) == (6, 6, 1.0)
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["best_step"] == 6
    assert rows[3]["best_val_ce"] == 2.0
    assert not list(output.glob("*.tmp"))


def test_legacy_checkpoint_starts_new_selection_history(run_setup, monkeypatch):
    model_cfg, training_cfg, paths = run_setup
    monkeypatch.setattr(loop, "evaluate", lambda *a, **kw: {"val_ce": 2.0})
    loop.train(model_cfg, training_cfg, max_steps=1, **paths)
    output = paths["output_dir"]
    legacy = torch.load(output / "last.pt", weights_only=True)
    del legacy["best_val_ce"], legacy["best_step"]
    torch.save(legacy, output / "last.pt")
    monkeypatch.setattr(loop, "evaluate", lambda *a, **kw: {"val_ce": 3.0})
    loop.train(model_cfg, training_cfg, max_steps=2, resume=output / "last.pt", **paths)
    best = torch.load(output / "best.pt", weights_only=True)
    assert (best["best_step"], best["best_val_ce"]) == (2, 3.0)


@pytest.mark.parametrize("invalid_ce", [float("nan"), float("inf")])
def test_nonfinite_validation_does_not_replace_best(run_setup, monkeypatch, invalid_ce):
    model_cfg, training_cfg, paths = run_setup
    monkeypatch.setattr(loop, "evaluate", lambda *a, **kw: {"val_ce": 2.0})
    loop.train(model_cfg, training_cfg, max_steps=1, **paths)
    output = paths["output_dir"]
    original_best = (output / "best.pt").read_bytes()
    monkeypatch.setattr(loop, "evaluate", lambda *a, **kw: {"val_ce": invalid_ce})
    with pytest.raises(RuntimeError, match="Non-finite validation"):
        loop.train(model_cfg, training_cfg, max_steps=2, resume=output / "last.pt", **paths)
    assert (output / "best.pt").read_bytes() == original_best
