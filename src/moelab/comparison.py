"""Sequential, isolated dense/MoE runs with bounded steps and per-run timeout."""

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from moelab.config import load_model_config, load_training_config
from moelab.model.budget import compare_budgets
from moelab.tokenizer.bpe import BPETokenizer


def fingerprint(path):
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_comparison(
    *,
    data_dir,
    output_dir,
    dense_path,
    moe_path,
    training_path,
    steps=20,
    seeds=(42, 43),
    device="cuda",
    threads=2,
    timeout_seconds=180,
):
    root = Path(__file__).resolve().parents[2]
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    training = load_training_config(training_path)
    if not 1 <= steps <= training.max_steps:
        raise ValueError("steps must be within the configured training horizon")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if timeout_seconds <= 0 or threads < 1:
        raise ValueError("Timeout and thread count must be positive")
    tokenizer = BPETokenizer.load(data / "tokenizer.json")
    models = {"dense": load_model_config(dense_path), "moe": load_model_config(moe_path)}
    for cfg in models.values():
        cfg.vocab_size = tokenizer.vocab_size
    budget = compare_budgets(models["dense"], models["moe"])
    if not budget["within_0_1_percent"]:
        raise ValueError("Active parameter gap exceeds 0.1%")
    hashes = {
        name: fingerprint(data / name)
        for name in ("ids_train.bin", "ids_val.bin", "tokenizer.json")
    }
    # Store source fingerprints even when the implementation is not committed yet.
    source_paths = sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py")])
    source_hashes = {p.relative_to(root).as_posix(): fingerprint(p) for p in source_paths}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    report = {
        "format_version": 1,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
        "source_sha256": source_hashes,
        "data_hashes": hashes,
        "budget": budget,
        "steps_per_run": steps,
        "seeds": list(seeds),
        "device_requested": device,
        "timeout_seconds_per_run": timeout_seconds,
        "model_configs": {k: v.model_dump() for k, v in models.items()},
        "training_config": training.model_dump(),
        "order_policy": "alternate dense-first and moe-first by seed index",
        "runs": [],
    }
    metadata = data / "metadata.json"
    if metadata.exists():
        report["data_metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "comparison.json"
    write_report(report_path, report)
    try:
        for index, seed in enumerate(seeds):
            for kind in ("dense", "moe") if index % 2 == 0 else ("moe", "dense"):
                run_dir = output / f"{kind}-seed-{seed}"
                run_dir.mkdir()
                model_path, config_path = run_dir / "model.yaml", run_dir / "training.yaml"
                model_path.write_text(yaml.safe_dump(models[kind].model_dump()), encoding="utf-8")
                config = training.model_copy(update={"seed": seed})
                config_path.write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")
                command = [
                    sys.executable,
                    str(root / "scripts/train.py"),
                    "--model",
                    str(model_path),
                    "--config",
                    str(config_path),
                    "--data-dir",
                    str(data),
                    "--output-dir",
                    str(run_dir),
                    "--device",
                    device,
                    "--max-steps",
                    str(steps),
                    "--threads",
                    str(threads),
                ]
                print(f"Running {kind}, seed {seed}, {steps} steps", flush=True)
                started = time.perf_counter()
                with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=timeout_seconds,
                    )
                process_seconds = time.perf_counter() - started
                summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
                if summary["data_hashes"] != hashes or summary["final_step"] != steps:
                    raise ValueError("Completed run does not match comparison data/step budget")
                report["runs"].append(
                    {"model": kind, "seed": seed, "process_seconds": process_seconds, **summary}
                )
                write_report(report_path, report)
        if any(fingerprint(data / name) != digest for name, digest in hashes.items()):
            raise ValueError("Data changed during comparison")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(report_path, report)
        raise
    report["finished_utc"] = datetime.now(UTC).isoformat()
    write_report(report_path, report)
    return report
