"""Run a bounded local comparison; no cloud resources are created."""

import argparse

from moelab.comparison import run_comparison

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True, help="New directory for logs and checkpoints")
    p.add_argument("--dense", default="configs/model/compare-dense.yaml")
    p.add_argument("--moe", default="configs/model/compare-moe.yaml")
    p.add_argument("--config", default="configs/training/laptop.yaml")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=180)
    args = p.parse_args()
    report = run_comparison(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dense_path=args.dense,
        moe_path=args.moe,
        training_path=args.config,
        steps=args.steps,
        seeds=args.seeds,
        device=args.device,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )
    for run in report["runs"]:
        print(
            f"{run['model']} seed={run['seed']} CE={run['final_val_ce']:.4f} "
            f"loop={run['loop_seconds']:.2f}s peak={run['peak_allocated_mib']} MiB"
        )
