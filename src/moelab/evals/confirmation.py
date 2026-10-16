"""Confirmation-only evaluation of the predeclared 0.01 versus 0.1 candidates."""

import json
import statistics
from pathlib import Path

import numpy as np
import torch

from moelab.comparison import fingerprint, write_report
from moelab.evals.full_stream import evaluate_full_stream
from moelab.training.loop import load_checkpoint


def evaluate_confirmation(
    ablation_dir, heldout_dir, output_dir, device="cuda", batch_size=2, max_tokens=200000
):
    source, data, output = map(Path, (ablation_dir, heldout_dir, output_dir))
    ablation = json.loads((source / "ablation.json").read_text(encoding="utf-8"))
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    if ablation["status"] != "complete":
        raise ValueError("A completed ablation is required")
    if metadata["training_data_hashes"] != ablation["data_hashes"]:
        raise ValueError("Confirmation preparation refers to different training data")
    for name, digest in metadata["artifacts_sha256"].items():
        if fingerprint(data / name) != digest:
            raise ValueError("Confirmation artifact fingerprint mismatch")
    if fingerprint(data / "tokenizer.json") != ablation["data_hashes"]["tokenizer.json"]:
        raise ValueError("Confirmation tokenizer must be frozen")
    size = (data / "ids_confirmation.bin").stat().st_size
    if batch_size < 1 or size % 2 or not 2 <= size // 2 <= max_tokens:
        raise ValueError("Invalid batch size or confirmation stream size")
    ids = np.memmap(data / "ids_confirmation.bin", dtype="<u2", mode="r")
    selected = [r for r in ablation["runs"] if r["coefficient"] in (0.01, 0.1)]
    expected = {(c, s) for c in (0.01, 0.1) for s in ablation["seeds"]}
    if (
        not expected
        or len(selected) != len(expected)
        or {(r["coefficient"], r["seed"]) for r in selected} != expected
    ):
        raise ValueError("Expected both predeclared candidates for every seed")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format_version": 1,
        "status": "running",
        "coefficients": [0.01, 0.1],
        "checkpoint_selection": "final ablation weights; no retraining or reselection",
        "ablation_sha256": fingerprint(source / "ablation.json"),
        "confirmation_metadata": metadata,
        "confirmation_metadata_sha256": fingerprint(data / "metadata.json"),
        "evaluation_source_sha256": fingerprint(Path(__file__)),
        "scoring_source_sha256": fingerprint(Path(__file__).with_name("full_stream.py")),
        "device": device,
        "max_tokens_guard": max_tokens,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "runs": [],
    }
    path = output / "confirmation.json"
    write_report(path, report)
    try:
        for run in selected:
            checkpoint = source / run["run_name"] / "last.pt"
            if fingerprint(checkpoint) != run["full_validation"]["checkpoint_sha256"]:
                raise ValueError("Checkpoint differs from the scored ablation checkpoint")
            model, saved = load_checkpoint(checkpoint, device)
            if (
                saved["step"] != ablation["steps"]
                or saved["data_hashes"] != ablation["data_hashes"]
                or saved["model_config"] != run["training_summary"]["model_config"]
                or saved["training_config"] != run["training_summary"]["training_config"]
            ):
                raise ValueError("Checkpoint does not match ablation metadata")
            del saved
            print(f"Confirming coefficient {run['coefficient']} seed {run['seed']}", flush=True)
            result = evaluate_full_stream(model, ids, batch_size)
            report["runs"].append(
                {
                    "coefficient": run["coefficient"],
                    "seed": run["seed"],
                    "checkpoint_sha256": fingerprint(checkpoint),
                    "development_full_ce": run["full_validation"]["val_ce"],
                    **result,
                }
            )
            del model
            write_report(path, report)
        for name, digest in metadata["artifacts_sha256"].items():
            if fingerprint(data / name) != digest:
                raise ValueError("Confirmation artifacts changed during scoring")
        pairs = []
        for seed in ablation["seeds"]:
            scores = {r["coefficient"]: r["val_ce"] for r in report["runs"] if r["seed"] == seed}
            pairs.append({"seed": seed, "candidate_minus_baseline_ce": scores[0.1] - scores[0.01]})
        values = [r["candidate_minus_baseline_ce"] for r in pairs]
        report.update(
            status="complete",
            paired_differences=pairs,
            mean_paired_difference=statistics.mean(values),
            sample_sd_paired_difference=statistics.stdev(values) if len(values) > 1 else None,
        )
        write_report(path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(path, report)
        raise
    return report
