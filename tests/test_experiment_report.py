import json
import math

import pytest

from moelab.evals.experiment import render_experiment, summarize_experiment


@pytest.fixture
def experiment(tmp_path):
    runs = []
    for kind, final_ce in [("dense", 2.0), ("moe", 2.2)]:
        directory = tmp_path / f"{kind}-seed-42"
        directory.mkdir()
        rows = []
        for step in range(5):
            ce = final_ce + (4 - step) / 10
            row = {"step": step, "val_ce": ce, "val_perplexity": math.exp(ce)}
            if step and kind == "moe":
                row["expert_counts"] = {"11": [1, 1, 1, 1] if step <= 2 else [2, 2, 0, 0]}
            rows.append(row)
        (directory / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        runs.append(
            dict(
                model=kind,
                seed=42,
                final_val_ce=final_ce,
                best_val_ce=final_ce,
                best_step=4,
                loop_seconds=1,
                loop_tokens_per_second=128,
                peak_allocated_mib=None,
                session_training_tokens=128,
            )
        )
    manifest = dict(status="complete", steps_per_run=4, seeds=[42], runs=runs)
    (tmp_path / "comparison.json").write_text(json.dumps(manifest))
    return tmp_path


def test_endpoints_paired_delta_and_windows(experiment):
    report = summarize_experiment(experiment, window_steps=2)
    assert report["early_window"] == [1, 2] and report["late_window"] == [3, 4]
    assert report["mean_paired_final_ce_difference"] == pytest.approx(0.2)
    assert report["sample_sd_paired_final_ce_difference"] is None
    dense, moe = report["runs"]
    assert dense["early_val_ce"] == 2.2
    assert dense["ce_change_from_early"] == pytest.approx(-0.2)
    assert len(dense["validation_curve"]) == 5
    assert moe["routing_early"]["layers"]["11"]["assignment_percentages"] == [25] * 4
    assert moe["routing_late"]["layers"]["11"]["assignment_percentages"] == [50, 50, 0, 0]
    assert moe["routing_late"]["layers"]["11"]["zero_assignment_steps_per_expert"] == [0, 0, 2, 2]
    text = render_experiment(report)
    assert "| 42 | +0.2000 |" in text
    assert "N/A" in text
    assert "50.00, 50.00, 0.00, 0.00" in text


@pytest.mark.parametrize("window", [0, 3])
def test_overlapping_or_empty_windows_fail(experiment, window):
    with pytest.raises(ValueError, match="nonoverlapping"):
        summarize_experiment(experiment, window)


@pytest.mark.parametrize("change", ["failed", "missing_pair", "unequal_tokens", "changed_ce"])
def test_incomplete_or_inconsistent_manifest_fails(experiment, change):
    path = experiment / "comparison.json"
    manifest = json.loads(path.read_text())
    if change == "failed":
        manifest["status"] = "failed"
    elif change == "missing_pair":
        manifest["runs"].pop()
    elif change == "unequal_tokens":
        manifest["runs"][0]["session_training_tokens"] += 1
    else:
        manifest["runs"][0]["final_val_ce"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        summarize_experiment(experiment, 2)


def test_missing_endpoint_or_step_is_not_silently_interpolated(experiment):
    path = experiment / "dense-seed-42/metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[2]["val_ce"]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="endpoint"):
        summarize_experiment(experiment, 2)
    rows.pop(2)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="every training step"):
        summarize_experiment(experiment, 2)
