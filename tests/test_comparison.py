import json
import subprocess

import numpy as np
import pytest
import yaml

from moelab.comparison import run_comparison
from moelab.config import ModelConfig, TrainingConfig
from moelab.tokenizer.bpe import BPETokenizer


@pytest.fixture
def comparison_args(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    BPETokenizer().save(data / "tokenizer.json")
    for split in ("train", "val"):
        np.tile(np.arange(40, dtype="<u2"), 10).tofile(data / f"ids_{split}.bin")
    moe = ModelConfig(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        ffn_hidden_dim=341,
        moe={"n_experts": 4, "top_k": 2},
    )
    dense = moe.model_dump() | {"moe": None, "ffn_hidden_dim": 683}
    training = TrainingConfig(
        lr=0.003, warmup_steps=0, max_steps=2, batch_size=2, precision="fp32", eval_every=1
    )
    for name, config in [
        ("dense", dense),
        ("moe", moe.model_dump()),
        ("training", training.model_dump()),
    ]:
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(config))
    return dict(
        data_dir=data,
        output_dir=tmp_path / "comparison",
        dense_path=tmp_path / "dense.yaml",
        moe_path=tmp_path / "moe.yaml",
        training_path=tmp_path / "training.yaml",
        steps=1,
        seeds=(42,),
        device="cpu",
        timeout_seconds=60,
    )


def test_isolated_comparison_records_equal_budgets_and_cpu_null_memory(comparison_args):
    pytest.importorskip("torch")
    result = run_comparison(**comparison_args)
    assert result["status"] == "complete"
    assert [r["model"] for r in result["runs"]] == ["dense", "moe"]
    a, b = result["runs"]
    assert a["data_hashes"] == b["data_hashes"] == result["data_hashes"]
    assert a["session_training_tokens"] == b["session_training_tokens"]
    for run in result["runs"]:
        assert run["final_step"] == 1 and run["best_step"] == 1
        assert run["loop_seconds"] > 0 and run["process_seconds"] > run["loop_seconds"]
        assert run["loop_tokens_per_second"] == pytest.approx(
            run["session_training_tokens"] / run["loop_seconds"]
        )
        assert run["peak_allocated_mib"] is None and run["peak_reserved_mib"] is None
        assert np.isfinite(run["final_val_ce"])
        assert run["evaluation"]["includes_aux_loss"] is False
    with pytest.raises(FileExistsError):
        run_comparison(**comparison_args)


def test_failed_worker_preserves_failed_manifest(comparison_args, monkeypatch):
    original = subprocess.run

    def fail(command, **kwargs):
        if "--max-steps" in command:
            raise subprocess.TimeoutExpired(command, 1)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        run_comparison(**comparison_args)
    report = json.loads((comparison_args["output_dir"] / "comparison.json").read_text())
    assert report["status"] == "failed" and report["runs"] == []
    assert "TimeoutExpired" in report["error"]


@pytest.mark.parametrize("override", [{"steps": 0}, {"seeds": (42, 42)}, {"timeout_seconds": 0}])
def test_invalid_budget_creates_no_output(comparison_args, override):
    with pytest.raises(ValueError):
        run_comparison(**(comparison_args | override))
    assert not comparison_args["output_dir"].exists()
