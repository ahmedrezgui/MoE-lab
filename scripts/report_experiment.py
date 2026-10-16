"""Report completed comparison runs with early/late routing and validation."""

import argparse
import json
from pathlib import Path

from moelab.evals.experiment import render_experiment, summarize_experiment

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-dir", required=True)
    p.add_argument("--output-dir", required=True, help="New report directory")
    p.add_argument("--window-steps", type=int, default=20)
    args = p.parse_args()
    report = summarize_experiment(args.comparison_dir, args.window_steps)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "analysis.md").write_text(render_experiment(report), encoding="utf-8")
    print(f"Wrote {output / 'analysis.md'} and analysis.json")
