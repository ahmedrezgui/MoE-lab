"""Archive a completed balancing sweep and render quality/routing comparisons."""

import argparse
import json
from pathlib import Path

from moelab.ablation import summarize_ablation
from moelab.comparison import fingerprint


def render_report(report):
    if report["status"] != "complete":
        raise ValueError("A completed sweep is required")
    summary = summarize_ablation(report["runs"], report["coefficients"], report["seeds"])
    if summary != report["coefficient_summary"]:
        raise ValueError("Stored paired summary differs from run results")
    lines = [
        "# Balancing-loss coefficient ablation",
        "",
        "Final checkpoints use the same data, MoE architecture, seeds and training budget. "
        "Only the load-balancing coefficient changes within a seed. "
        "Full validation uses consecutive context-reset blocks, including the short tail.",
        "",
        "| Coefficient | Mean full CE | Mean paired CE difference vs 0.01 "
        "| Sample SD of difference |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        sd = row["sample_sd_paired_difference"]
        sd_text = f"{sd:.4f}" if sd is not None else "N/A"
        lines.append(
            f"| {row['coefficient']:g} | {row['mean_full_ce']:.4f} | "
            f"{row['mean_paired_difference']:+.4f} | {sd_text} |"
        )
    lines += [
        "",
        "Negative paired CE differences favor the displayed coefficient over 0.01. "
        "Seed spread is descriptive, not a significance test. Raw training objectives "
        "are not comparable because their auxiliary terms have different weights.",
        "",
        "| Coefficient | Seed | Full CE | Sampled final CE | Early zero events "
        "| Late zero events | Loop seconds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted(report["runs"], key=lambda r: (r["coefficient"], r["seed"])):
        zeros = {
            window: sum(
                sum(layer["zero_assignment_steps_per_expert"])
                for layer in run[f"routing_{window}"]["layers"].values()
            )
            for window in ("early", "late")
        }
        lines.append(
            f"| {run['coefficient']:g} | {run['seed']} | {run['full_validation']['val_ce']:.4f} | "
            f"{run['training_summary']['final_val_ce']:.4f} | {zeros['early']} | "
            f"{zeros['late']} | {run['training_summary']['loop_seconds']:.2f} |"
        )
    lines += [
        "",
        "A zero event is one expert receiving no assignments in one optimizer step, "
        "summed across layers and steps in the window. It is not a count of "
        "permanently dead experts. "
        "Early/late windows are steps 1 through window_steps and the final window_steps; "
        f"here window_steps={report['window_steps']} and final step={report['steps']}.",
        "",
        "## Per-layer routing shares",
        "",
        "Shares count assignments, not unique tokens or gate weights. Layer IDs are logged "
        "module IDs, not block indices. Vectors follow zero-based expert order. "
        "Max/uniform is 1 for equal shares; it can reach 2 for four experts with top-2.",
        "",
        "| Coefficient | Seed | Layer | Window | Assignment shares (%) "
        "| Max/uniform | Zero-step counts |",
        "| ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for run in sorted(report["runs"], key=lambda r: (r["coefficient"], r["seed"])):
        for window in ("early", "late"):
            for layer_id, layer in run[f"routing_{window}"]["layers"].items():
                shares = ", ".join(f"{p:.2f}" for p in layer["assignment_percentages"])
                lines.append(
                    f"| {run['coefficient']:g} | {run['seed']} | {layer_id} | {window} | "
                    f"{shares} | {layer['max_to_uniform_ratio']:.3f} | "
                    f"{layer['zero_assignment_steps_per_expert']} |"
                )
    lines += [
        "",
        "The accompanying ablation.json preserves per-step window counts, validation curves, "
        "full validation target counts, checkpoint/source/data hashes and resolved configurations. "
        "Raw metric logs are archived by run name. Balanced routing does not imply semantic "
        "specialization or better prediction quality. These validation data have been used "
        "during development; choosing a coefficient here needs confirmation on new held-out data.",
        "",
    ]
    return "\n".join(lines)


def archive_report(ablation_dir, output_dir):
    source, output = Path(ablation_dir), Path(output_dir)
    report = json.loads((source / "ablation.json").read_text(encoding="utf-8"))
    text = render_report(report)
    for run in report["runs"]:
        if fingerprint(source / run["run_name"] / "metrics.jsonl") != run["metrics_sha256"]:
            raise ValueError("Metrics fingerprint mismatch")
    output.mkdir(parents=True, exist_ok=False)
    (output / "ablation.json").write_bytes((source / "ablation.json").read_bytes())
    for run in report["runs"]:
        directory = output / run["run_name"]
        directory.mkdir()
        (directory / "metrics.jsonl").write_bytes(
            (source / run["run_name"] / "metrics.jsonl").read_bytes()
        )
    (output / "results.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation-dir", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    archive_report(a.ablation_dir, a.output_dir)
