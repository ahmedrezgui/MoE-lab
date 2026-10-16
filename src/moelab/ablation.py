"""Controlled load-balancing coefficient sweep with isolated train/eval workers."""

import json
import math
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from moelab.comparison import fingerprint, write_report
from moelab.config import ModelConfig, TrainingConfig, load_model_config, load_training_config
from moelab.evals.routing import summarize_routing
from moelab.tokenizer.bpe import BPETokenizer


def run_worker(job_path):
    import numpy as np
    import torch

    from moelab.evals.full_stream import evaluate_full_stream
    from moelab.training.loop import train

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    torch.set_num_threads(job["threads"])
    model = train(
        ModelConfig.model_validate(job["model_config"]),
        TrainingConfig.model_validate(job["training_config"]),
        job["steps"],
        data_dir=job["data_dir"],
        output_dir=job["output_dir"],
        device=job["device"],
    )
    # train returns final weights after saving last.pt, not the selected best weights.
    ids = np.memmap(Path(job["data_dir"]) / "ids_val.bin", dtype="<u2", mode="r")
    result = evaluate_full_stream(model, ids, job["eval_batch_size"])
    result["checkpoint_sha256"] = fingerprint(Path(job["output_dir"]) / "last.pt")
    write_report(Path(job["output_dir"]) / "full-validation.json", result)


def summarize_ablation(runs, coefficients, seeds, baseline=0.01):
    expected = {(coef, seed) for coef in coefficients for seed in seeds}
    keys = [(r["coefficient"], r["seed"]) for r in runs]
    if len(keys) != len(expected) or set(keys) != expected or baseline not in coefficients:
        raise ValueError("Need exactly one run per coefficient/seed and the baseline")
    baselines = {
        r["seed"]: r["full_validation"]["val_ce"] for r in runs if r["coefficient"] == baseline
    }
    summary = []
    for coef in coefficients:
        selected = [r for r in runs if r["coefficient"] == coef]
        differences = [r["full_validation"]["val_ce"] - baselines[r["seed"]] for r in selected]
        summary.append(
            {
                "coefficient": coef,
                "mean_full_ce": statistics.mean(r["full_validation"]["val_ce"] for r in selected),
                "paired_differences_vs_baseline": [
                    {
                        "seed": r["seed"],
                        "ce_difference": r["full_validation"]["val_ce"] - baselines[r["seed"]],
                    }
                    for r in selected
                ],
                "mean_paired_difference": statistics.mean(differences),
                "sample_sd_paired_difference": statistics.stdev(differences)
                if len(differences) > 1
                else None,
            }
        )
    return summary


def run_ablation(
    *,
    data_dir,
    output_dir,
    model_path,
    training_path,
    coefficients=(0.0, 0.01, 0.1),
    seeds=(42, 43, 44),
    steps=200,
    window_steps=20,
    device="cuda",
    threads=2,
    timeout_seconds=180,
    max_val_tokens=100000,
):
    root = Path(__file__).resolve().parents[2]
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    cfg, training = load_model_config(model_path), load_training_config(training_path)
    if cfg.moe is None:
        raise ValueError("A MoE model is required")
    if (
        not coefficients
        or len(set(coefficients)) != len(coefficients)
        or any(not math.isfinite(c) or c < 0 for c in coefficients)
        or 0.01 not in coefficients
    ):
        raise ValueError("Use distinct finite nonnegative coefficients including baseline 0.01")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Use distinct nonnegative integer seeds")
    if not 1 <= steps <= training.max_steps or not 1 <= window_steps <= steps // 2:
        raise ValueError("Invalid steps or overlapping routing windows")
    if threads < 1 or timeout_seconds <= 0 or max_val_tokens < 2:
        raise ValueError("Invalid execution bounds")
    size = (data / "ids_val.bin").stat().st_size
    if size % 2 or not 2 <= size // 2 <= max_val_tokens:
        raise ValueError("Validation token count exceeds bounds or has incomplete IDs")
    cfg.vocab_size = BPETokenizer.load(data / "tokenizer.json").vocab_size
    hashes = {
        name: fingerprint(data / name)
        for name in ("ids_train.bin", "ids_val.bin", "tokenizer.json")
    }
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    report = {
        "format_version": 1,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_sha256": {
            p.relative_to(root).as_posix(): fingerprint(p)
            for p in sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py")])
        },
        "base_model_config": cfg.model_dump(),
        "base_training_config": training.model_dump(),
        "data_hashes": hashes,
        "coefficients": list(coefficients),
        "seeds": list(seeds),
        "baseline_coefficient": 0.01,
        "steps": steps,
        "window_steps": window_steps,
        "timeout_seconds_per_worker": timeout_seconds,
        "max_val_tokens": max_val_tokens,
        "order_policy": "cyclic coefficient order rotated by seed index",
        "selection_policy": "fresh training; final step weights, no best-checkpoint reselection",
        "runs": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "ablation.json"
    write_report(manifest_path, report)
    try:
        for index, seed in enumerate(seeds):
            order = list(coefficients[index % len(coefficients) :]) + list(
                coefficients[: index % len(coefficients)]
            )
            for coef in order:
                variant = cfg.model_copy(deep=True)
                variant.moe.load_balance_coef = coef
                train_cfg = training.model_copy(update={"seed": seed})
                name = f"coef-{coef:g}-seed-{seed}"
                directory = output / name
                directory.mkdir()
                job = {
                    "model_config": variant.model_dump(),
                    "training_config": train_cfg.model_dump(),
                    "steps": steps,
                    "data_dir": str(data),
                    "output_dir": str(directory),
                    "device": device,
                    "threads": threads,
                    "eval_batch_size": 2,
                }
                write_report(directory / "job.json", job)
                print(f"Training and evaluating {name}", flush=True)
                start = time.perf_counter()
                with (directory / "console.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(root / "scripts/run_balance_ablation.py"),
                            "--worker-job",
                            str(directory / "job.json"),
                        ],
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=timeout_seconds,
                    )
                duration = time.perf_counter() - start
                summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
                full = json.loads((directory / "full-validation.json").read_text(encoding="utf-8"))
                if (
                    summary["data_hashes"] != hashes
                    or summary["final_step"] != steps
                    or summary["model_config"] != job["model_config"]
                    or summary["training_config"] != job["training_config"]
                    or full["scored_targets"] != size // 2 - 1
                ):
                    raise ValueError("Worker output does not match declared experiment")
                metrics = directory / "metrics.jsonl"
                rows = [
                    json.loads(line) for line in metrics.read_text().splitlines() if line.strip()
                ]
                if [r["step"] for r in rows] != list(range(steps + 1)):
                    raise ValueError("Missing or repeated training steps")
                report["runs"].append(
                    {
                        "run_name": name,
                        "coefficient": coef,
                        "seed": seed,
                        "process_seconds": duration,
                        "training_summary": summary,
                        "full_validation": full,
                        "initial_val_ce": rows[0]["val_ce"],
                        "metrics_sha256": fingerprint(metrics),
                        "validation_curve": [
                            {k: r[k] for k in ("step", "val_ce", "val_perplexity")}
                            for r in rows
                            if "val_ce" in r
                        ],
                        "routing_early": summarize_routing(
                            metrics, start_step=1, end_step=window_steps
                        ),
                        "routing_late": summarize_routing(
                            metrics, start_step=steps - window_steps + 1, end_step=steps
                        ),
                    }
                )
                write_report(manifest_path, report)
        if any(fingerprint(data / name) != digest for name, digest in hashes.items()):
            raise ValueError("Data changed during sweep")
        report.update(
            status="complete",
            finished_utc=datetime.now(UTC).isoformat(),
            coefficient_summary=summarize_ablation(report["runs"], coefficients, seeds),
        )
        write_report(manifest_path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(manifest_path, report)
        raise
    return report
