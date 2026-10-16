# Local GPU smoke comparison

This is a bounded pipeline experiment on an RTX 3050 Ti Laptop GPU (4 GB), using PyTorch 2.11.0+cu128, bf16 training and fp32 validation. The [machine-readable record](local-gpu-smoke.json) contains exact configurations, data/source SHA-256 fingerprints, software environment, run order, measurement scopes, and all results. The recorded Git HEAD predates this uncommitted implementation; `git_dirty` is true and source fingerprints identify the code actually used.

## Protocol

- Same TinyStories prefix: 8 training documents and 4 held-out validation documents. Exact duplicates/overlap are removed during preparation. This is not a representative benchmark sample.
- BPE fitted on 128 training bytes; requested capacity 512, actual vocabulary 272, including EOS ID 271. Token streams contain 5,091 training and 2,541 validation tokens.
- Backbone: width 128, two layers, context 128. Dense hidden width 683; MoE four experts of width 341, top-2, balancing coefficient 0.01. Active counts: 658,304 dense and 658,560 MoE (256-weight gap). Total MoE: 1,182,336.
- Each run: 20 optimizer steps, batch 2, accumulation 4, 20,480 processed training targets. The learning-rate schedule still has the configured 200-step horizon and 10-step warmup.
- Paired seeds 42 and 43; order dense/MoE then MoE/dense. Each model runs in a separate process with a 180-second timeout. No resume or previous weights used.
- Validation uses four sampled batches, two sequences each, context 128: 1,024 prediction targets sampled with fixed seed 123, with possible overlap. It is not full-corpus validation. CE excludes auxiliary loss. Final step 20 is compared at equal token budgets. Best step is also 20 for every run because it is the only post-training evaluation in these runs.

## Recorded measurements

| Model | Seed | Final CE | Perplexity | Loop seconds | Process seconds | Loop tokens/s | Allocated MiB | Reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 42 | 3.4256 | 30.74 | 1.739 | 7.030 | 11,776 | 33.63 | 38.00 |
| moe | 42 | 3.4805 | 32.48 | 2.946 | 7.987 | 6,951 | 43.82 | 48.00 |
| moe | 43 | 3.4660 | 32.01 | 2.892 | 7.766 | 7,082 | 43.82 | 48.00 |
| dense | 43 | 3.4326 | 30.96 | 1.373 | 6.259 | 14,917 | 33.63 | 38.00 |

Loop duration includes training, final validation, logging, and checkpoint writes, and excludes model setup and initial validation. CUDA is synchronized at its start and end. It includes first-step optimizer allocation and kernel warmup; there is no separate steady-state timing warmup. Process duration additionally includes interpreter startup, imports, initialization, initial validation, summary writing, and shutdown. Throughput is processed training targets divided by loop duration, not GPU-kernel-only throughput.

Allocated memory is PyTorch's peak live tensor allocation. Reserved memory includes the caching allocator's pool. Peaks reset immediately before the loop; resident model weights and setup-created cache remain. These values exclude CUDA context/driver memory, other applications, and allocations outside PyTorch, and do not represent total GPU usage. No peak system-RAM measurement is made.

## Interpretation

Dense has lower final validation CE, shorter measured loop time, and lower allocated memory for both paired seeds here. This supports the ability to run and measure both architectures locally. It does not establish a general dense/MoE ranking: the corpus and training duration are tiny, there are only two seeds, timing is short and subject to laptop clocks/background activity, and deterministic CUDA algorithms are not enforced. Alternating order reduces one simple ordering bias but does not eliminate thermal or cache effects. No statistical significance claim or cloud-cost comparison is made.

## Reproduce

From the repository root, with the development environment installed, use fresh artifact paths:

```powershell
.venv/Scripts/python scripts/download_data.py --output data/raw/gpu-compare --train-documents 8 --val-documents 4
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/gpu-compare/train.jsonl --val-input data/raw/gpu-compare/val.jsonl --output data/gpu-compare --block-size 128 --vocab-size 512 --tokenizer-training-bytes 128 --add-eos
.venv/Scripts/python scripts/run_comparison.py --data-dir data/gpu-compare --output-dir checkpoints/gpu-compare --steps 20 --seeds 42 43 --device cuda --timeout-seconds 180
```

The downloader uses the repository's pinned TinyStories revision. Keep its manifest, source card and license notice with local artifacts. Changed output/source path metadata and the preparation block-size check can differ from the original smoke preparation, while the token stream and tokenizer hashes should match. The comparison runner derives both model vocabularies from the actual tokenizer without editing checked-in presets. New timings and CUDA numerical results need not be identical.

Original local artifacts are under `checkpoints/comparison-gpu-smoke/`: per-run YAML configs, console logs, metrics, summaries, best/last checkpoints, and the comparison manifest. Checkpoints and corpus files remain ignored; this small result record is intended for Git. Next experiments should increase corpus coverage and duration, use more paired seeds, and inspect expert utilization before drawing architecture conclusions.
