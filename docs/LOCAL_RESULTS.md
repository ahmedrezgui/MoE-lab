# Local verification report

Date: 2026-10-15. Environment: Windows, Python 3.12.8, PyTorch 2.11.0+cu128, NVIDIA RTX 3050 Ti Laptop GPU (4096 MiB), bf16 supported.

## Verification

- 21 tests passed, including dense/MoE overfitting, causal independence, sparse routing versus a dense reference, router gradients, BPE Unicode round trips, dataset boundaries, exact CPU resume equivalence with dropout/accumulation, and API readiness/generation.
- Ruff lint and formatting passed. `pip check` reported no broken requirements.
- One upstream Starlette/httpx TestClient deprecation warning remains; tests succeed.
- CPU training and checkpoint generation succeeded. CUDA bf16 training succeeded, stopping at step 20 and resuming to step 200.
- The trained GPU checkpoint also loaded and served successfully on CPU through FastAPI TestClient. No public server was deployed.
- Docker build and remote GitHub Actions execution have not been verified.

## GPU toy experiment

Configuration: `configs/model/laptop-moe.yaml`, `configs/training/laptop.yaml`.

- 1,180,288 total parameters; 656,512 estimated active parameters.
- Byte vocabulary 256; 1,643 training tokens and 183 validation tokens from `examples/demo.txt`.
- Corpus hash: d2c56478a2aaa716afdc76a842d56b9190cbb7b645c9600492e509abcca2fafd.
- 200 optimizer steps, 204,800 sampled training tokens (many repeated exposures).

| Step | Validation CE | Validation perplexity |
|---|---:|---:|
| 0 | 5.5516 | 257.6450 |
| 20 | 3.0150 | 20.3898 |
| 60 | 2.2849 | 9.8251 |
| 100 | 2.2175 | 9.1848 |
| 200 | 2.5717 | 13.0877 |

The model overfits: training loss keeps decreasing while validation CE worsens after approximately step 100. This is an intentionally tiny plumbing test; overlapping sampled validation windows and a single short story do not provide a reliable generalization estimate. The trainer currently keeps the latest checkpoint, not the best validation checkpoint.

The resumed 180-step session took approximately 24.6 seconds through final evaluation, around 7,495 sampled tokens/sec including loop/evaluation overhead. This excludes startup and final checkpoint serialization. Peak PyTorch allocated tensor memory was 43.73 MiB; this is NOT total GPU process/driver/reserved memory. These are single-run observations, not a standardized benchmark.

Example seeded stochastic continuation at step 200:

> Once upon a time, a child fox near the rabbit opened the tall tree. The bird shared the seed the tmaater stone beside the ltst the flowe

It has learned local spelling/story patterns but still makes errors and incoherent combinations. No claims of conversational ability are warranted.

Local artifacts (ignored by Git): `checkpoints/laptop/last.pt`, `checkpoints/laptop/tokenizer.json`, `checkpoints/laptop/metrics.jsonl`; CPU smoke artifacts are in `checkpoints/cpu-smoke/`.

## Next experiment

Use a documented, document-separated real corpus. Preserve a fixed evaluation set, add best-checkpoint selection, and compare routing utilization and held-out CE across dense/MoE configurations with explicit budgets. Assess local throughput and memory at that workload before paying for cloud compute.
