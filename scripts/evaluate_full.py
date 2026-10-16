"""Evaluate all final checkpoints in a completed comparison on the full val stream."""

import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import torch

from moelab.comparison import fingerprint, write_report
from moelab.evals.full_stream import evaluate_full_stream
from moelab.tokenizer.bpe import BPETokenizer
from moelab.training.loop import load_checkpoint


def evaluate_comparison(
    comparison_dir, data_dir, output_dir, device="cuda", batch_size=2, max_tokens=1000000
):
    root, data, output = Path(comparison_dir), Path(data_dir), Path(output_dir)
    manifest = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Comparison must be complete")
    if batch_size < 1 or max_tokens < 2:
        raise ValueError("Positive batch size and at least two max_tokens required")
    val_path = data / "ids_val.bin"
    size = val_path.stat().st_size
    if size % 2 or not 2 <= size // 2 <= max_tokens:
        raise ValueError("Validation file must contain 2..max_tokens complete uint16 IDs")
    hashes = {name: fingerprint(data / name) for name in ("ids_val.bin", "tokenizer.json")}
    if any(manifest["data_hashes"][name] != digest for name, digest in hashes.items()):
        raise ValueError("Validation/tokenizer fingerprints differ from training")
    tokenizer = BPETokenizer.load(data / "tokenizer.json")
    ids = np.memmap(val_path, dtype="<u2", mode="r")
    pairs = {(run["model"], run["seed"]) for run in manifest["runs"]}
    expected = {(kind, seed) for seed in manifest["seeds"] for kind in ("dense", "moe")}
    if not expected or pairs != expected or len(manifest["runs"]) != len(expected):
        raise ValueError("Expected exactly one dense/MoE pair per seed")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format_version": 1,
        "status": "running",
        "checkpoint_selection": "last.pt only",
        "comparison_sha256": fingerprint(root / "comparison.json"),
        "data_hashes": hashes,
        "device": device,
        "max_tokens_guard": max_tokens,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "threads": torch.get_num_threads(),
        },
        "evaluation_source_sha256": fingerprint(Path(__file__)),
        "scoring_source_sha256": fingerprint(
            Path(__file__).parents[1] / "src/moelab/evals/full_stream.py"
        ),
        "runs": [],
    }
    path = output / "full-validation.json"
    write_report(path, report)
    try:
        for run in manifest["runs"]:
            checkpoint = root / f"{run['model']}-seed-{run['seed']}" / "last.pt"
            print(f"Evaluating {run['model']} seed {run['seed']}", flush=True)
            model, saved = load_checkpoint(checkpoint, device)
            if (
                saved["step"] != manifest["steps_per_run"]
                or model.cfg.model_dump() != run["model_config"]
                or saved["training_config"] != run["training_config"]
                or saved["data_hashes"] != manifest["data_hashes"]
                or tokenizer.vocab_size != model.cfg.vocab_size
            ):
                raise ValueError("Checkpoint config/step/data do not match comparison")
            if "gpu" not in report["environment"]:
                report["environment"]["gpu"] = (
                    torch.cuda.get_device_name(next(model.parameters()).device)
                    if str(device).startswith("cuda")
                    else None
                )
            # Checkpoint optimizer tensors remain on CPU; release before forward scoring.
            del saved
            result = evaluate_full_stream(model, ids, batch_size)
            report["runs"].append(
                {
                    "model": run["model"],
                    "seed": run["seed"],
                    "step": manifest["steps_per_run"],
                    "checkpoint_sha256": fingerprint(checkpoint),
                    "previous_sampled_val_ce": run["final_val_ce"],
                    **result,
                }
            )
            del model
            write_report(path, report)
        if any(fingerprint(data / name) != digest for name, digest in hashes.items()):
            raise ValueError("Evaluation input changed during scoring")
        deltas = []
        for seed in manifest["seeds"]:
            pair = {r["model"]: r["val_ce"] for r in report["runs"] if r["seed"] == seed}
            deltas.append({"seed": seed, "full_ce_moe_minus_dense": pair["moe"] - pair["dense"]})
        values = [r["full_ce_moe_minus_dense"] for r in deltas]
        report.update(
            status="complete",
            paired_differences=deltas,
            mean_paired_difference=statistics.mean(values),
            sample_sd_paired_difference=statistics.stdev(values) if len(values) > 1 else None,
        )
        write_report(path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(path, report)
        raise
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1000000,
        help="Reject larger validation files; never truncate coverage",
    )
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    evaluate_comparison(
        args.comparison_dir,
        args.data_dir,
        args.output_dir,
        args.device,
        args.batch_size,
        args.max_tokens,
    )
