"""Protect the comparability checks before spending a training budget."""

import importlib.util
from pathlib import Path

import pytest

from moelab.comparison import fingerprint

spec = importlib.util.spec_from_file_location(
    "story_experiment", Path(__file__).parents[1] / "scripts/run_story_experiment.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def controls(tmp_path):
    for name in ("tokenizer.json", "ids_val.bin"):
        (tmp_path / name).write_bytes(b"original")
    saved = {
        "model_config": {"moe": {"load_balance_coef": 0.1}},
        "training_config": {"seed": 42, "lr": 0.001, "max_steps": 200},
        "data_hashes": {
            name: fingerprint(tmp_path / name) for name in ("tokenizer.json", "ids_val.bin")
        },
    }
    training = {**saved["training_config"], "max_steps": 2000}
    return saved, training, tmp_path


def test_allows_longer_horizon(controls):
    saved, training, data = controls
    module.validate_controls(saved, saved["model_config"], training, data)


@pytest.mark.parametrize("name", ["tokenizer.json", "ids_val.bin"])
def test_rejects_changed_tokenizer_or_validation(controls, name):
    saved, training, data = controls
    (data / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="must match exactly"):
        module.validate_controls(saved, saved["model_config"], training, data)


def test_rejects_changed_architecture(controls):
    saved, training, data = controls
    with pytest.raises(ValueError, match="architecture"):
        module.validate_controls(saved, {"moe": {"load_balance_coef": 0.01}}, training, data)


@pytest.mark.parametrize("key,value", [("seed", 43), ("lr", 0.01)])
def test_rejects_unplanned_training_changes(controls, key, value):
    saved, training, data = controls
    training[key] = value
    with pytest.raises(ValueError, match="Training control changed"):
        module.validate_controls(saved, saved["model_config"], training, data)
