"""Train from token files prepared with scripts/prepare_data.py."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import torch

from moelab.config import load_model_config, load_training_config
from moelab.training.loop import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/training.yaml")
    p.add_argument("--model", default="configs/moe.yaml")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--data-dir", default="data/tinystories-5k-eos")
    p.add_argument("--output-dir", default="checkpoints/moe")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--resume")
    p.add_argument(
        "--restart",
        action="store_true",
        help="Archive the current checkpoints/moe directory and train fresh",
    )
    args = p.parse_args()
    model_cfg = load_model_config(args.model)
    training_cfg = load_training_config(args.config)
    if args.restart:
        if args.resume:
            p.error("Use either --restart or --resume, not both")
        root = Path("checkpoints").resolve()
        output = Path(args.output_dir).resolve()
        if output != root / "moe":
            p.error("--restart is only supported for checkpoints/moe")
        if output.exists():
            archive = root / "archive" / ("moe-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
            if not archive.resolve().is_relative_to(root):
                p.error("Archive must stay within checkpoints")
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
            print(f"Previous run preserved in {archive}")
    torch.set_num_threads(args.threads)
    train(
        model_cfg,
        training_cfg,
        args.max_steps,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
