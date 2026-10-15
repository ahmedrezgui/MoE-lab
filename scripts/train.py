"""Train from token files prepared with scripts/prepare_data.py."""

import argparse

import torch

from moelab.config import load_model_config, load_training_config
from moelab.training.loop import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--data-dir", default="data/demo")
    p.add_argument("--output-dir", default="checkpoints/demo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--resume")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    train(
        load_model_config(args.model),
        load_training_config(args.config),
        args.max_steps,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
