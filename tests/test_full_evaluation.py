import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from moelab.evals.full_stream import evaluate_full_stream


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.cfg = SimpleNamespace(max_seq_len=4, vocab_size=256)
        self.inputs = []

    def forward(self, x):
        self.inputs.extend(x.tolist())
        logits = torch.zeros(*x.shape, 256, device=x.device)
        logits[..., 0] = x.float() / 5
        return logits, None


@pytest.mark.parametrize("length", [2, 5, 11, 13])
def test_complete_target_coverage_tail_weighting_and_batch_invariance(length):
    ids = np.arange(length, dtype=np.uint16)
    model = RecordingModel()
    result = evaluate_full_stream(model, ids, batch_size=2)
    expected = sum(math.log(math.exp(i / 5) + 255) for i in range(length - 1)) / (length - 1)
    assert result["val_ce"] == pytest.approx(expected, rel=1e-6)
    assert result["scored_targets"] == length - 1
    assert result["tail_targets"] == (length - 1) % 4
    assert result["full_blocks"] == (length - 1) // 4
    # Each input predicts its next stream position; no dropped or duplicate targets.
    assert [x + 1 for block in model.inputs for x in block] == list(range(1, length))
    assert model.training
    model.eval()
    other = evaluate_full_stream(model, ids, batch_size=1)
    assert not model.training
    assert other["val_ce"] == pytest.approx(result["val_ce"], abs=1e-6)
    assert result["evaluation_seconds"] > 0


@pytest.mark.parametrize(
    "ids",
    [
        np.array([0]),
        np.array([[0, 1]]),
        np.array([-1, 0]),
        np.array([0, 256]),
        np.array([0.0, 1.0]),
    ],
)
def test_invalid_streams(ids):
    with pytest.raises(ValueError):
        evaluate_full_stream(RecordingModel(), ids)


def test_forward_failure_restores_mode():
    class Failing(RecordingModel):
        def forward(self, x):
            raise RuntimeError("test failure")

    model = Failing()
    with pytest.raises(RuntimeError, match="test failure"):
        evaluate_full_stream(model, np.array([0, 1]))
    assert model.training
    with pytest.raises(ValueError, match="fp32"):
        evaluate_full_stream(RecordingModel().double(), np.array([0, 1]))
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_full_stream(RecordingModel(), np.array([0, 1]), 0)


def test_checkpoint_evaluation_checks_provenance_and_preserves_outputs(tmp_path):
    from scripts.evaluate_full import evaluate_comparison

    from moelab.comparison import fingerprint
    from moelab.config import ModelConfig
    from moelab.model.transformer import Transformer
    from moelab.tokenizer.bpe import BPETokenizer

    torch.set_num_threads(2)
    data, comparison = tmp_path / "data", tmp_path / "comparison"
    data.mkdir()
    comparison.mkdir()
    BPETokenizer().save(data / "tokenizer.json")
    np.arange(35, dtype="<u2").tofile(data / "ids_val.bin")
    hashes = {name: fingerprint(data / name) for name in ("ids_val.bin", "tokenizer.json")}
    hashes["ids_train.bin"] = "unused-training-fingerprint"
    runs = []
    for kind in ("dense", "moe"):
        cfg = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            moe={"n_experts": 2, "top_k": 1} if kind == "moe" else None,
        )
        directory = comparison / f"{kind}-seed-42"
        directory.mkdir()
        torch.save(
            {
                "model_config": cfg.model_dump(),
                "model": Transformer(cfg).state_dict(),
                "step": 2,
                "training_config": {"seed": 42},
                "data_hashes": hashes,
            },
            directory / "last.pt",
        )
        runs.append(
            {
                "model": kind,
                "seed": 42,
                "model_config": cfg.model_dump(),
                "training_config": {"seed": 42},
                "final_val_ce": 5.5,
            }
        )
    (comparison / "comparison.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "steps_per_run": 2,
                "seeds": [42],
                "runs": runs,
                "data_hashes": hashes,
            }
        )
    )
    output = tmp_path / "report"
    report = evaluate_comparison(comparison, data, output, device="cpu")
    assert report["status"] == "complete"
    assert len(report["runs"]) == 2
    assert all(r["scored_targets"] == 34 and r["tail_targets"] == 2 for r in report["runs"])
    assert report["sample_sd_paired_difference"] is None
    with pytest.raises(FileExistsError):
        evaluate_comparison(comparison, data, output, device="cpu")
    with pytest.raises(ValueError, match="max_tokens"):
        evaluate_comparison(comparison, data, tmp_path / "over-budget", device="cpu", max_tokens=10)
    np.arange(36, dtype="<u2").tofile(data / "ids_val.bin")
    with pytest.raises(ValueError, match="fingerprints"):
        evaluate_comparison(comparison, data, tmp_path / "wrong-data", device="cpu")
    assert not (tmp_path / "wrong-data").exists()
