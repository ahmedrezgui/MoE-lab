"""Thin MLflow wrapper so the rest of the code never imports mlflow directly.

Working from day one — start `make mlflow` and log something today.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


@contextmanager
def start_run(experiment: str, run_name: str | None, params: dict[str, Any]):
    """Usage:
    with start_run("moelab", None, {**model_cfg.model_dump(), **train_cfg.model_dump()}) as log:
        log({"train/loss": loss}, step=step)
    """
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_flatten(params))

        def log(metrics: dict[str, float], step: int) -> None:
            mlflow.log_metrics(metrics, step=step)

        yield log


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
