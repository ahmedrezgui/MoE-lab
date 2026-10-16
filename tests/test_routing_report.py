import json

import pytest

from moelab.evals.routing import render_routing_markdown, summarize_routing


def write_metrics(tmp_path, rows):
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_weighted_aggregation_and_step_concentration(tmp_path):
    path = write_metrics(
        tmp_path,
        [
            {"step": 0, "val_ce": 5},
            {"step": 1, "expert_counts": {"11": [1, 1], "38": [2, 0]}},
            {"step": 2, "expert_counts": {"11": [6, 0], "38": [0, 6]}},
        ],
    )
    report = summarize_routing(path)
    layer = report["layers"]["11"]
    assert layer["counts"] == [7, 1]
    assert layer["assignment_percentages"] == [87.5, 12.5]  # Not the mean 75/25.
    assert layer["max_to_uniform_ratio"] == 1.75
    assert layer["peak_step_max_to_uniform_ratio"] == 2
    assert layer["zero_assignment_steps_per_expert"] == [0, 1]
    assert layer["per_step"][0]["assignment_percentages"] == [50, 50]
    assert layer["zero_assignment_experts"] == []
    assert report["observed_steps"] == 2
    assert len(report["metrics_sha256"]) == 64
    window = summarize_routing(path, start_step=2, end_step=2)
    assert window["layers"]["11"]["zero_assignment_experts"] == [1]
    markdown = render_routing_markdown([report])
    assert "| 11 | 0 | 7 | 87.50 | 0 |" in markdown
    assert "not decoder block" in markdown
    assert "not percentages of unique tokens" in markdown


@pytest.mark.parametrize(
    "counts",
    [[0, 0], [-1, 2], [1.5, 2], [True, 2], [float("nan"), 1], [float("inf"), 1], [1], "bad"],
)
def test_invalid_counts_rejected_with_line_number(tmp_path, counts):
    path = write_metrics(tmp_path, [{"step": 1, "expert_counts": {"11": counts}}])
    with pytest.raises(ValueError, match="metrics.jsonl:1:"):
        summarize_routing(path)


@pytest.mark.parametrize(
    "second",
    [
        {"step": 1, "expert_counts": {"11": [1, 1]}},
        {"step": 2, "expert_counts": {"38": [1, 1]}},
        {"step": 2, "expert_counts": {"11": [1, 1, 1]}},
        {"step": 3, "expert_counts": {"11": [1, 1]}},
        {"step": 2, "train_loss": 1, "expert_counts": {}},
    ],
)
def test_inconsistent_or_missing_observations_rejected(tmp_path, second):
    path = write_metrics(tmp_path, [{"step": 1, "expert_counts": {"11": [1, 1]}}, second])
    with pytest.raises(ValueError):
        summarize_routing(path)


def test_dense_empty_and_invalid_windows(tmp_path):
    path = write_metrics(tmp_path, [{"step": 1, "train_loss": 1, "expert_counts": {}}])
    with pytest.raises(ValueError, match="dense"):
        summarize_routing(path)
    path = write_metrics(tmp_path, [{"step": 0, "val_ce": 1}])
    with pytest.raises(ValueError, match="No MoE"):
        summarize_routing(path)
    with pytest.raises(ValueError, match="start_step"):
        summarize_routing(path, start_step=0)
    with pytest.raises(ValueError, match="end_step"):
        summarize_routing(path, start_step=5, end_step=4)


def test_layer_totals_must_agree(tmp_path):
    path = write_metrics(tmp_path, [{"step": 1, "expert_counts": {"11": [1, 1], "38": [1, 2]}}])
    with pytest.raises(ValueError, match="totals"):
        summarize_routing(path)
