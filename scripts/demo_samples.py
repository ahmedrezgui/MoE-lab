"""Record all fixed prompts/settings for the portfolio demo without cherry-picking."""

import argparse
import json
from pathlib import Path

import torch

from moelab.serving.demo import DEFAULT_CHECKPOINT, EXAMPLE_PROMPTS, DemoModel

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        p.error("Output exists; choose a new file.")
    torch.set_num_threads(2)
    engine = DemoModel(a.checkpoint, a.device)
    samples = []
    for prompt in EXAMPLE_PROMPTS:
        for temperature in (0.0, 0.8):
            samples.append(
                {"prompt": prompt, **engine.complete(prompt, 64, temperature, 40, 0.95, 42)}
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"checkpoint": engine.info, "samples": samples}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved all {len(samples)} fixed samples to {output}")
