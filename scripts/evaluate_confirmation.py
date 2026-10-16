"""Score the frozen 0.01/0.1 checkpoints on the reserved confirmation stream."""

import argparse

import torch

from moelab.evals.confirmation import evaluate_confirmation

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation-dir", required=True)
    p.add_argument("--heldout-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=200000)
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(a.threads)
    evaluate_confirmation(
        a.ablation_dir, a.heldout_dir, a.output_dir, a.device, a.batch_size, a.max_tokens
    )
