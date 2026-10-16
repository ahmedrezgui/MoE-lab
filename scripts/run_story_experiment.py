"""Bounded fresh training, same-stream baseline scoring, and fixed sample export."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from moelab.comparison import fingerprint, write_report
from moelab.config import load_model_config, load_training_config
from moelab.evals.full_stream import evaluate_full_stream
from moelab.evals.routing import summarize_routing
from moelab.training.loop import load_checkpoint


def validate_controls(saved, model_config, training_config, data):
    if saved["model_config"] != model_config:
        raise ValueError("Baseline architecture/balancing coefficient must match")
    for name in ("tokenizer.json", "ids_val.bin"):
        if saved["data_hashes"][name] != fingerprint(data / name):
            raise ValueError(f"Baseline {name} must match exactly")
    allowed_changes = {"max_steps", "eval_every", "checkpoint_every"}
    for key, value in training_config.items():
        if key not in allowed_changes and saved["training_config"][key] != value:
            raise ValueError(f"Training control changed: {key}")


def run(args):
    root = Path(__file__).resolve().parents[1]
    data, output, report_dir = map(Path, (args.data_dir, args.output_dir, args.report_dir))
    if args.timeout_seconds <= 0 or args.threads < 1:
        raise ValueError("Timeout and threads must be positive")
    if output.exists() or report_dir.exists():
        raise ValueError("Use new checkpoint and report directories")
    size = (data / "ids_val.bin").stat().st_size
    if size % 2 or not 2 <= size // 2 <= 100000:
        raise ValueError("Expected 2..100000 complete validation tokens")
    cfg = load_model_config(args.model).model_dump()
    training = load_training_config(args.config).model_dump()
    torch.set_num_threads(args.threads)
    baseline_model, saved = load_checkpoint(args.baseline, "cpu")
    validate_controls(saved, cfg, training, data)
    baseline_step = saved["step"]
    del saved
    hashes = {
        name: fingerprint(data / name)
        for name in ("tokenizer.json", "ids_train.bin", "ids_val.bin")
    }
    output.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    report = {
        "status": "running",
        "selection": "fresh seed-42 training; final weights; baseline fixed before run",
        "interpretation": "joint data and training-budget change; not a single-variable ablation",
        "timeout_seconds_training_and_evaluation": args.timeout_seconds,
        "baseline_checkpoint_sha256": fingerprint(args.baseline),
        "baseline_step": baseline_step,
        "data_hashes": hashes,
        "model_config": cfg,
        "training_config": training,
        "source_sha256": {
            p.relative_to(root).as_posix(): fingerprint(p)
            for p in sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py")])
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
    }
    path = report_dir / "experiment.json"
    write_report(path, report)
    try:
        # CPU fp32 baseline avoids keeping another model resident on the training GPU.
        ids = np.memmap(data / "ids_val.bin", dtype="<u2", mode="r")
        report["baseline_full_validation"] = evaluate_full_stream(baseline_model, ids, 2)
        report["baseline_evaluation_device"] = "cpu"
        del baseline_model, ids
        job = {
            "model_config": cfg,
            "training_config": training,
            "steps": training["max_steps"],
            "data_dir": str(data.resolve()),
            "output_dir": str(output.resolve()),
            "device": args.device,
            "threads": args.threads,
            "eval_batch_size": 2,
        }
        write_report(output / "job.json", job)
        write_report(path, report)
        start = time.perf_counter()
        with (output / "console.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/run_balance_ablation.py"),
                    "--worker-job",
                    str((output / "job.json").resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout_seconds,
                cwd=root,
            )
        report["worker_seconds"] = time.perf_counter() - start
        summary = json.loads((output / "run_summary.json").read_text())
        full = json.loads((output / "full-validation.json").read_text())
        if (
            summary["data_hashes"] != hashes
            or summary["model_config"] != cfg
            or summary["training_config"] != training
            or summary["final_step"] != training["max_steps"]
            or full["scored_targets"] != size // 2 - 1
        ):
            raise ValueError("Worker result differs from declared experiment")
        if any(fingerprint(data / name) != digest for name, digest in hashes.items()):
            raise ValueError("Training/evaluation data changed during experiment")
        report["training_summary"] = summary
        report["candidate_full_validation"] = full
        report["candidate_evaluation_device"] = args.device
        # Re-score candidate on CPU too, so the headline comparison uses one device.
        candidate, saved = load_checkpoint(output / "last.pt", "cpu")
        del saved
        ids = np.memmap(data / "ids_val.bin", dtype="<u2", mode="r")
        report["candidate_cpu_full_validation"] = evaluate_full_stream(candidate, ids, 2)
        del candidate, ids
        report["cpu_ce_difference"] = (
            report["candidate_cpu_full_validation"]["val_ce"]
            - report["baseline_full_validation"]["val_ce"]
        )
        metrics = output / "metrics.jsonl"
        window = min(100, training["max_steps"] // 2)
        report["routing_early"] = summarize_routing(metrics, start_step=1, end_step=window)
        report["routing_late"] = summarize_routing(
            metrics,
            start_step=training["max_steps"] - window + 1,
            end_step=training["max_steps"],
        )
        for name in ("metrics.jsonl", "run_summary.json", "full-validation.json"):
            shutil.copyfile(output / name, report_dir / name)
        shutil.copyfile(data / "metadata.json", report_dir / "data-metadata.json")
        for label, checkpoint in (("before", args.baseline), ("after", output / "last.pt")):
            subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/demo_samples.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                    "--output",
                    str(report_dir / f"samples-{label}.json"),
                ],
                check=True,
                timeout=120,
            )
        report["status"] = "complete"
        write_report(path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(path, report)
        raise
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/tinystories-5k-eos")
    p.add_argument("--output-dir", default="checkpoints/story-5k-2k")
    p.add_argument("--report-dir", default="reports/story-5k-2k")
    p.add_argument("--baseline", default="checkpoints/balance-ablation-1k/coef-0.1-seed-42/last.pt")
    p.add_argument("--model", default="configs/model/portfolio-moe.yaml")
    p.add_argument("--config", default="configs/training/story-2k.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=900)
    run(p.parse_args())
