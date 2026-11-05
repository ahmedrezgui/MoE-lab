"""The single-model launcher preserves old runs without starting GPU jobs in tests."""

import sys
from pathlib import Path

import pytest
from scripts import train as launcher

ROOT = Path(__file__).resolve().parents[1]


def arguments(monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--model",
            str(ROOT / "configs/moe.yaml"),
            "--config",
            str(ROOT / "configs/training.yaml"),
            *extra,
        ],
    )


def test_restart_preserves_previous_run_and_uses_single_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = tmp_path / "checkpoints/moe"
    previous.mkdir(parents=True)
    (previous / "last.pt").write_bytes(b"existing weights")
    calls = []
    monkeypatch.setattr(launcher, "train", lambda *args, **kwargs: calls.append((args, kwargs)))
    arguments(monkeypatch, "--restart")
    launcher.main()
    archives = list((tmp_path / "checkpoints/archive").glob("moe-*/last.pt"))
    assert len(archives) == 1 and archives[0].read_bytes() == b"existing weights"
    assert not previous.exists()
    args, kwargs = calls[0]
    assert args[0].moe.n_experts == 4 and args[1].max_steps == 20000
    assert kwargs["output_dir"] == "checkpoints/moe"


@pytest.mark.parametrize(
    "extra",
    [
        ["--resume", "checkpoints/moe/last.pt"],
        ["--output-dir", "checkpoints/other"],
    ],
)
def test_restart_rejects_ambiguous_or_noncanonical_paths(tmp_path, monkeypatch, extra):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "train", lambda *a, **kw: pytest.fail("Training must not run"))
    arguments(monkeypatch, "--restart", *extra)
    with pytest.raises(SystemExit):
        launcher.main()
