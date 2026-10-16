"""Summarize existing MoE metrics into JSON and Markdown; no training required."""

import argparse
import json
from pathlib import Path

from moelab.evals.routing import render_routing_markdown, summarize_routing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, help="New report directory")
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    args = parser.parse_args()
    reports = [
        summarize_routing(path, start_step=args.start_step, end_step=args.end_step)
        for path in args.metrics
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "routing.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "routing.md").write_text(render_routing_markdown(reports), encoding="utf-8")
    print(f"Wrote {output / 'routing.md'} and routing.json")
