"""Compare validation trajectories and preselected early/late routing windows."""

import json
import statistics
from pathlib import Path

from moelab.comparison import fingerprint
from moelab.evals.routing import summarize_routing


def summarize_experiment(comparison_dir, window_steps=20):
    root = Path(comparison_dir)
    manifest_path = root / "comparison.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_step = manifest["steps_per_run"]
    if manifest["status"] != "complete":
        raise ValueError("Comparison must be complete")
    if window_steps < 1 or 2 * window_steps > final_step:
        raise ValueError("Require positive, nonoverlapping early and late windows")
    results = []
    for run in manifest["runs"]:
        path = root / f"{run['model']}-seed-{run['seed']}" / "metrics.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [row["step"] for row in rows] != list(range(final_step + 1)):
            raise ValueError("Expected step zero and every training step exactly once")
        if rows[-1].get("val_ce") != run["final_val_ce"]:
            raise ValueError("Final validation does not match comparison summary")
        if "val_ce" not in rows[window_steps]:
            raise ValueError("Early window endpoint must have a recorded validation evaluation")
        result = {
            "model": run["model"],
            "seed": run["seed"],
            "metrics_sha256": fingerprint(path),
            "validation_curve": [
                {
                    "step": row["step"],
                    "val_ce": row["val_ce"],
                    "val_perplexity": row["val_perplexity"],
                }
                for row in rows
                if "val_ce" in row
            ],
            "early_val_ce": rows[window_steps]["val_ce"],
            "final_val_ce": run["final_val_ce"],
            "ce_change_from_early": run["final_val_ce"] - rows[window_steps]["val_ce"],
            "best_val_ce": run["best_val_ce"],
            "best_step": run["best_step"],
            "loop_seconds": run["loop_seconds"],
            "loop_tokens_per_second": run["loop_tokens_per_second"],
            "peak_allocated_mib": run["peak_allocated_mib"],
            "session_training_tokens": run["session_training_tokens"],
        }
        if run["model"] == "moe":
            result["routing_early"] = summarize_routing(path, start_step=1, end_step=window_steps)
            result["routing_late"] = summarize_routing(
                path, start_step=final_step - window_steps + 1, end_step=final_step
            )
        results.append(result)
    pairs = []
    for seed in manifest["seeds"]:
        selected = [run for run in results if run["seed"] == seed]
        if len(selected) != 2 or {run["model"] for run in selected} != {"dense", "moe"}:
            raise ValueError("Expected one dense/MoE pair per seed")
        dense = next(run for run in selected if run["model"] == "dense")
        moe = next(run for run in selected if run["model"] == "moe")
        if dense["session_training_tokens"] != moe["session_training_tokens"]:
            raise ValueError("Paired runs must have equal token budgets")
        pairs.append(
            {"seed": seed, "final_ce_moe_minus_dense": moe["final_val_ce"] - dense["final_val_ce"]}
        )
    if len(results) != 2 * len(manifest["seeds"]) or not pairs:
        raise ValueError("Unexpected runs or empty seed list")
    deltas = [pair["final_ce_moe_minus_dense"] for pair in pairs]
    return {
        "format_version": 1,
        "comparison_sha256": fingerprint(manifest_path),
        "early_window": [1, window_steps],
        "late_window": [final_step - window_steps + 1, final_step],
        "runs": results,
        "paired_final_ce_differences": pairs,
        "mean_paired_final_ce_difference": statistics.mean(deltas),
        "sample_sd_paired_final_ce_difference": statistics.stdev(deltas)
        if len(deltas) > 1
        else None,
        "interpretation": "Negative MoE-minus-dense CE favors MoE on this sampled evaluation; "
        "routing windows describe training assignments, not validation specialization. "
        "Seed spread is descriptive, not a significance test.",
    }


def render_experiment(report):
    lines = [
        "# Longer bounded comparison: validation and routing",
        "",
        f"Early routing: steps {report['early_window']}. "
        f"Late routing: steps {report['late_window']}. "
        "CE is measured at each window's final step, not averaged over its training losses.",
        "",
        "| Model | Seed | Early CE | Final CE | Change | Best CE (step) | "
        "Loop seconds | Peak allocated MiB |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for run in report["runs"]:
        memory = run["peak_allocated_mib"]
        memory_text = f"{memory:.2f}" if memory is not None else "N/A"
        lines.append(
            f"| {run['model']} | {run['seed']} | {run['early_val_ce']:.4f} | "
            f"{run['final_val_ce']:.4f} | {run['ce_change_from_early']:+.4f} | "
            f"{run['best_val_ce']:.4f} ({run['best_step']}) | "
            f"{run['loop_seconds']:.2f} | {memory_text} |"
        )
    lines += [
        "",
        "## Paired final validation differences",
        "",
        "Positive MoE-minus-dense CE favors dense; negative favors MoE.",
        "",
        "| Seed | MoE minus dense CE |",
        "| --- | ---: |",
    ]
    for pair in report["paired_final_ce_differences"]:
        lines.append(f"| {pair['seed']} | {pair['final_ce_moe_minus_dense']:+.4f} |")
    lines += [
        "",
        f"Mean paired difference: {report['mean_paired_final_ce_difference']:+.4f}. "
        f"Sample standard deviation: {report['sample_sd_paired_final_ce_difference']}. "
        "This spread is descriptive, not a significance test.",
        "",
        "## Early and late expert assignments",
        "",
        "Shares are percentages of hard assignments (top-2 gives two per token). "
        "Layer IDs are logged module IDs. Expert order in each vector is 0, 1, 2, 3. "
        "Max/uniform = 1 means equal traffic. Zero-step counts are per expert within "
        "the window, not the number of experts unused for the entire run.",
        "",
        "| Seed | Layer | Window | Expert shares (%) | Max/uniform | Zero-step counts |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for run in report["runs"]:
        if run["model"] != "moe":
            continue
        for window in ("early", "late"):
            for layer_id, layer in run[f"routing_{window}"]["layers"].items():
                shares = ", ".join(f"{p:.2f}" for p in layer["assignment_percentages"])
                lines.append(
                    f"| {run['seed']} | {layer_id} | {window} | {shares} | "
                    f"{layer['max_to_uniform_ratio']:.3f} | "
                    f"{layer['zero_assignment_steps_per_expert']} |"
                )
    lines += [
        "",
        "The accompanying analysis JSON includes all recorded validation evaluations "
        "and per-step routing data for both windows. Aggregate traffic can hide transient "
        "concentration. Neither balanced traffic nor a CE change demonstrates semantic "
        "specialization or a causal benefit from balancing loss. Timing includes evaluation, "
        "logging and checkpoint I/O; memory covers the PyTorch allocator, not total GPU usage.",
        "",
    ]
    return "\n".join(lines)
