"""Render a static portfolio figure from archived observations (requires matplotlib)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / "reports/story-5k-2k"
    experiment = json.loads((report / "experiment.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (report / "metrics.jsonl").read_text().splitlines()]
    evaluated = [row for row in rows if "val_ce" in row]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    fig.suptitle("MoE Lab | Measured training progress", fontsize=18, weight="bold")
    axes[0].plot(
        [r["step"] for r in evaluated],
        [r["val_ce"] for r in evaluated],
        color="#007f86",
        linewidth=2.5,
    )
    axes[0].set(
        title="5,000 stories: sampled development loss",
        xlabel="Optimizer update",
        ylabel="Cross entropy (lower is better)",
        ylim=(0, 7),
    )
    axes[0].grid(alpha=0.2)
    values = [
        experiment[key]["val_ce"]
        for key in ("baseline_full_validation", "candidate_cpu_full_validation")
    ]
    bars = axes[1].bar(
        ["1,000 stories\n200 updates", "5,000 stories\n2,000 updates"],
        values,
        color=["#60758a", "#007f86"],
        width=0.55,
    )
    axes[1].bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=5)
    axes[1].set(
        title="Full development loss: same 28,667 targets",
        ylabel="Cross entropy, CPU fp32",
        ylim=(0, 5),
    )
    fig.supxlabel(
        "One seed; data and training budget change together. Sampled and full scoring differ.",
        fontsize=10,
    )
    output = ROOT / "assets"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "training-results.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
