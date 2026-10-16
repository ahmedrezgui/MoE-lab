import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from moelab.comparison import run_comparison
from moelab.tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def comparison_args(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    BPETokenizer().save(data / "tokenizer.json")
    for split in ("train", "val"):
        np.tile(np.arange(40, dtype="<u2"), 10).tofile(data / f"ids_{split}.bin")
    return dict(
        data_dir=data,
        output_dir=tmp_path / "comparison",
        dense_path=ROOT / "configs/model/compare-dense.yaml",
        moe_path=ROOT / "configs/model/compare-moe.yaml",
        training_path=ROOT / "configs/training/demo.yaml",
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
