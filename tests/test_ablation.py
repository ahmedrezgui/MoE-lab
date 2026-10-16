import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from moelab.ablation import run_ablation, summarize_ablation
from moelab.config import ModelConfig, TrainingConfig
from moelab.model.transformer import Transformer
from moelab.tokenizer.bpe import BPETokenizer


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    BPETokenizer().save(data / "tokenizer.json")
    for split in ("train", "val"):
        np.tile(np.arange(40, dtype="<u2"), 3).tofile(data / f"ids_{split}.bin")
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        ffn_hidden_dim=32,
        moe={"n_experts": 2, "top_k": 1},
    )
    training = TrainingConfig(
        lr=0.001,
        warmup_steps=0,
        max_steps=2,
        batch_size=1,
        precision="fp32",
        eval_every=1,
        checkpoint_every=2,
    )
    model, train = tmp_path / "model.yaml", tmp_path / "train.yaml"
    model.write_text(yaml.safe_dump(cfg.model_dump()))
    train.write_text(yaml.safe_dump(training.model_dump()))
    return dict(
        data_dir=data,
        output_dir=tmp_path / "sweep",
        model_path=model,
        training_path=train,
        steps=2,
        window_steps=1,
        seeds=(42,),
        device="cpu",
        timeout_seconds=60,
    )


def test_coefficient_does_not_change_initial_weights_or_logits():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        moe={"n_experts": 4, "top_k": 2},
    )
    states = []
    for coefficient in (0, 0.01, 0.1):
        cfg.moe.load_balance_coef = coefficient
        torch.manual_seed(42)
        model = Transformer(cfg).eval()
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    for other in states[1:]:
        for name, weight in states[0].items():
            torch.testing.assert_close(weight, other[name], atol=0, rtol=0)


def test_isolated_sweep_preserves_controls_and_scores_all_targets(setup):
    result = run_ablation(**setup)
    assert result["status"] == "complete" and len(result["runs"]) == 3
    assert len({run["initial_val_ce"] for run in result["runs"]}) == 1
    configs = []
    for run in result["runs"]:
        summary = run["training_summary"]
        cfg = summary["model_config"]
        assert cfg["moe"].pop("load_balance_coef") == run["coefficient"]
        configs.append(cfg)
        assert summary["session_training_tokens"] == 64
        assert run["full_validation"]["scored_targets"] == 119
        assert run["routing_early"]["first_step"] == 1
        assert run["routing_late"]["first_step"] == 2
        assert len(run["validation_curve"]) == 3
    assert configs[0] == configs[1] == configs[2]
    baseline = next(x for x in result["coefficient_summary"] if x["coefficient"] == 0.01)
    assert baseline["mean_paired_difference"] == 0
    with pytest.raises(FileExistsError):
        run_ablation(**setup)


@pytest.mark.parametrize(
    "override",
    [
        {"coefficients": (-1, 0.01)},
        {"coefficients": (0, 0.1)},
        {"coefficients": (0.01, 0.01)},
        {"coefficients": (float("nan"), 0.01)},
        {"seeds": (42, 42)},
        {"window_steps": 2},
    ],
)
def test_invalid_sweep_creates_no_output(setup, override):
    with pytest.raises(ValueError):
        run_ablation(**(setup | override))
    assert not setup["output_dir"].exists()


def test_paired_summary_uses_matching_seed_not_execution_order():
    runs = [
        {"coefficient": c, "seed": s, "full_validation": {"val_ce": v}}
        for c, s, v in [(0, 2, 5), (0.01, 1, 2), (0.01, 2, 4), (0, 1, 1)]
    ]
    result = summarize_ablation(runs, [0, 0.01], [1, 2])
    assert result[0]["mean_paired_difference"] == 0
    assert result[0]["sample_sd_paired_difference"] == pytest.approx(2**0.5)
    with pytest.raises(ValueError):
        summarize_ablation(runs[:-1], [0, 0.01], [1, 2])


def test_worker_failure_records_failed_manifest(setup, monkeypatch):
    import subprocess

    original = subprocess.run

    def fail(command, **kwargs):
        if "--worker-job" in command:
            raise subprocess.TimeoutExpired(command, 1)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        run_ablation(**setup)
    result = json.loads((Path(setup["output_dir"]) / "ablation.json").read_text())
    assert result["status"] == "failed" and result["runs"] == []
