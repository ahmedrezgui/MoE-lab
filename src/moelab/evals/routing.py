"""Report hard routing assignments from training JSONL, without loading weights."""

import hashlib
import json
import math
from pathlib import Path


def assignment_stats(counts):
    total = sum(counts)
    fractions = [count / total for count in counts]
    return {
        "assignments": total,
        "counts": counts,
        "assignment_percentages": [100 * fraction for fraction in fractions],
        "max_to_uniform_ratio": max(fractions) * len(counts),
        "zero_assignment_experts": [i for i, count in enumerate(counts) if count == 0],
    }


def summarize_routing(metrics_path, *, start_step=None, end_step=None):
    """Aggregate counts, not percentages; reject ambiguous or incomplete routing rows.

    Layer keys are logged module IDs, not decoder block indices. Metrics may contain
    evaluation-only rows without expert_counts, which do not contribute assignments.
    """
    if start_step is not None and start_step < 1:
        raise ValueError("start_step must be positive")
    if end_step is not None and (end_step < 1 or end_step < (start_step or 1)):
        raise ValueError("Invalid end_step")
    path = Path(metrics_path)
    raw = path.read_bytes()
    layers = {}
    previous = -1
    selected_steps = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            step = row["step"]
            if type(step) is not int or step < 0 or step <= previous:
                raise ValueError("Steps must be nonnegative, unique and increasing")
            previous = step
            if step == 0 or (start_step is not None and step < start_step):
                continue
            if end_step is not None and step > end_step:
                continue
            counts_by_layer = row.get("expert_counts")
            if not counts_by_layer:
                if "train_loss" in row:
                    raise ValueError(
                        "Training row has no routing counts (dense runs are unsupported)"
                    )
                continue
            if not isinstance(counts_by_layer, dict):
                raise ValueError("expert_counts must map layer IDs to count arrays")
            if layers and set(counts_by_layer) != set(layers):
                raise ValueError("Routing layer IDs changed")
            totals = []
            for layer_id, counts in counts_by_layer.items():
                if not isinstance(counts, list) or len(counts) < 2:
                    raise ValueError("Each layer needs at least two expert counts")
                if any(
                    type(n) not in (int, float) or not math.isfinite(n) or n < 0 or int(n) != n
                    for n in counts
                ):
                    raise ValueError("Counts must be finite nonnegative integers")
                counts = [int(n) for n in counts]
                if sum(counts) == 0:
                    raise ValueError("A routing layer has zero total assignments")
                layer = layers.setdefault(layer_id, {"total": [0] * len(counts), "steps": []})
                if len(counts) != len(layer["total"]):
                    raise ValueError("Expert count changed")
                layer["total"] = [a + b for a, b in zip(layer["total"], counts, strict=True)]
                layer["steps"].append({"step": step, **assignment_stats(counts)})
                totals.append(sum(counts))
            if len(set(totals)) != 1:
                raise ValueError("Layers disagree on assignment totals for a training step")
            selected_steps.append(step)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
    if not selected_steps:
        raise ValueError("No MoE routing observations in the selected step range")
    # A missing optimizer step would silently bias a window summary.
    if selected_steps != list(range(selected_steps[0], selected_steps[-1] + 1)):
        raise ValueError("Missing routing observations within selected steps")
    result = {
        "format_version": 1,
        "metrics_file": path.as_posix(),
        "metrics_sha256": hashlib.sha256(raw).hexdigest(),
        "requested_start_step": start_step,
        "requested_end_step": end_step,
        "first_step": selected_steps[0],
        "last_step": selected_steps[-1],
        "observed_steps": len(selected_steps),
        "scope": "training hard top-k assignments across all microbatches; "
        "percentages use assignments, not unique tokens or gate weights",
        "layer_id_convention": "logged model.modules() IDs, not decoder block indices",
        "layers": {},
    }
    for layer_id, layer in layers.items():
        result["layers"][layer_id] = {
            **assignment_stats(layer["total"]),
            "zero_assignment_steps_per_expert": [
                sum(step["counts"][i] == 0 for step in layer["steps"])
                for i in range(len(layer["total"]))
            ],
            "peak_step_max_to_uniform_ratio": max(
                step["max_to_uniform_ratio"] for step in layer["steps"]
            ),
            "per_step": layer["steps"],
        }
    return result


def render_routing_markdown(reports):
    lines = [
        "# Expert-utilization report",
        "",
        "Percentages count hard expert assignments during training, summed over the selected "
        "steps and microbatches. Each layer sums to 100%. With top-2, each token contributes "
        "two assignments; these are not percentages of unique tokens or weighted router outputs.",
        "",
        "Layer IDs are the recorded module IDs, not decoder block numbers. "
        "A max/uniform ratio of 1 means equal assignment shares. With four experts, 1.6 means "
        "the busiest expert receives 40% of assignments versus the uniform 25%. "
        "No automatic collapse threshold is imposed.",
        "",
    ]
    for report in reports:
        lines += [
            f"## {report['metrics_file']}",
            "",
            f"Observed steps {report['first_step']}–{report['last_step']} "
            f"({report['observed_steps']} steps). Source SHA-256: `{report['metrics_sha256']}`.",
            "",
            "| Layer ID | Expert | Assignments | Share (%) | Steps with zero assignments |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for layer_id, layer in report["layers"].items():
            for expert, count in enumerate(layer["counts"]):
                lines.append(
                    f"| {layer_id} | {expert} | {count:,} | "
                    f"{layer['assignment_percentages'][expert]:.2f} | "
                    f"{layer['zero_assignment_steps_per_expert'][expert]} |"
                )
        lines += [
            "",
            "| Layer ID | Aggregate max/uniform | Highest step max/uniform |",
            "| --- | ---: | ---: |",
        ]
        for layer_id, layer in report["layers"].items():
            lines.append(
                f"| {layer_id} | {layer['max_to_uniform_ratio']:.3f} | "
                f"{layer['peak_step_max_to_uniform_ratio']:.3f} |"
            )
        lines.append("")
    lines += [
        "Aggregate shares can hide changes over time. The accompanying JSON keeps every "
        "step's counts and percentages; use step windows to compare early and late routing. "
        "Zero assignments indicate no dispatched work in an observed step, not necessarily "
        "a dead expert outside this window. Balanced traffic does not prove useful "
        "specialization or better language quality. These counts do not measure validation "
        "routing or time spent by each expert.",
        "",
    ]
    return "\n".join(lines)
