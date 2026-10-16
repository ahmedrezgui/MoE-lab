"""Run a bounded load-balancing coefficient sweep on local hardware."""

import argparse

from moelab.ablation import run_ablation, run_worker

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker-job", help=argparse.SUPPRESS)
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--model", default="configs/model/compare-moe.yaml")
    p.add_argument("--config", default="configs/training/laptop.yaml")
    p.add_argument("--coefficients", type=float, nargs="+", default=[0, 0.01, 0.1])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--window-steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--timeout-seconds", type=int, default=180)
    p.add_argument("--max-val-tokens", type=int, default=100000)
    a = p.parse_args()
    if a.worker_job:
        run_worker(a.worker_job)
    else:
        if not a.data_dir or not a.output_dir:
            p.error("--data-dir and --output-dir are required")
        run_ablation(
            data_dir=a.data_dir,
            output_dir=a.output_dir,
            model_path=a.model,
            training_path=a.config,
            coefficients=a.coefficients,
            seeds=a.seeds,
            steps=a.steps,
            window_steps=a.window_steps,
            device=a.device,
            threads=a.threads,
            timeout_seconds=a.timeout_seconds,
            max_val_tokens=a.max_val_tokens,
        )
