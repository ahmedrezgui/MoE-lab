# TinyStories 1k: 200-step dense/MoE comparison

Six bounded local GPU runs completed: dense and MoE over paired seeds 42, 43 and 44. Both models improved from step 20 to step 200. Dense had lower final sampled validation CE in every pair, with mean 3.8940 versus 4.0062 for MoE. The mean MoE-minus-dense gap was +0.1122 (sample standard deviation 0.0143). This is a descriptive result from three seeds, not a statistical significance claim or general architecture ranking.

Read the [validation and early/late routing tables](analysis.md), [analysis JSON](analysis.json), and [complete comparison manifest](comparison.json). Raw per-step metrics for all six runs are archived in the model/seed subdirectories alongside this report. These small records are tracked; the corpus and large checkpoints remain ignored.

## Protocol chosen before observing results

- Source: first 1,000 training and 100 validation documents of pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, not a random corpus sample. See [download provenance](download-manifest.json). The source card declares CDLA-Sharing-1.0; original card/notice remain with local raw data.
- Exact normalization/deduplication and overlap removal retained all 1,000/100 documents. BPE fits 65,536 bytes from the training prefix only (89 sampled documents, final one truncated). Actual vocabulary 512 includes EOS ID 511. Data contains 412,068 training and 28,668 validation tokens; see [preparation metadata](preparation-metadata.json).
- Shared backbone: two layers, width 128, context 128, four query heads/two key-value heads. Dense FFN width 683; MoE four experts of width 341, top-2, balancing coefficient 0.01. Dense total/active weights 689,024; MoE active 689,280, total 1,213,056. The active gap is 256 (about 0.037% of dense).
- Same laptop training config: AdamW, peak learning rate 0.001, 10-step warmup, 200-step schedule, batch 2, accumulation 4, bf16 training, gradient clipping 1.0, weight decay 0.01. Every model processes 204,800 prediction targets; block sampling uses replacement and is not one exhaustive pass through the corpus.
- Seeds 42, 43 and 44; order dense/MoE, MoE/dense, dense/MoE. Each worker is a separate process with a 180-second timeout. The machine is an RTX 3050 Ti Laptop GPU, roughly 4 GB VRAM, PyTorch 2.11.0+cu128. No cloud resources used.
- Validation at step zero and every 20 steps uses four batches of two sequences at context 128 with seed 123: 1,024 sampled held-out targets, potentially overlapping. It runs in fp32 without auxiliary balancing loss. The early/late CE comparison uses snapshots at steps 20/200.
- Routing windows fixed before results: steps 1-20 and 181-200, each 40,960 top-2 assignments per layer. All experts' shares and zero-assignment steps are reported; windows were not chosen to favor an observed trend.

## Findings and limits

Dense mean loop duration was 12.45 seconds versus 24.92 seconds for MoE. Peak allocated memory was approximately 34.98 MiB versus 45.17 MiB. Loop time includes validation, logging and checkpoint writes; it excludes setup/initial validation and is CUDA-synchronized at boundaries. Memory is PyTorch allocator peak, not total driver/device usage. Short laptop timings vary with clocks and background activity; the manifest preserves individual runs and process times.

Every expert receives assignments in both aggregate windows. For seed 44, logged layer 38 experts 0 and 1 received no assignments in 1 and 2 early steps, respectively. No expert has a zero-assignment step in any late window. However, balance does not improve uniformly: seed 42 layer 38's max/uniform ratio rises from 1.039 to 1.206 while its validation CE improves. Seed 43's ratios decline in both layers. These observations demonstrate that traffic concentration and prediction loss are distinct measurements; they do not identify semantic specialization or prove that balancing loss caused learning.

All six best checkpoints were at step 200, the end of this run. That does not imply 200 is an optimal stopping point. The models are still tiny, the corpus is a selected prefix, validation covers only a fixed sample, and only three seeds were tested. The mean difference and sample spread are descriptive, not confidence intervals or significance tests. Expert IDs do not represent equivalent learned functions across different random initializations.

Do not compare these CE/perplexity values directly to the earlier 8/4-document smoke run: its tokenizer had 272 entries and this one has 512. A changed tokenizer changes the units of token-level loss. Within this experiment, the data/tokenizer fingerprints are shared across both models and all seeds.

## Reproduce training

Use fresh directories, or change the paths below if the artifacts already exist:

```powershell
.venv/Scripts/python scripts/download_data.py --output data/raw/tinystories-1k --train-documents 1000 --val-documents 100
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/tinystories-1k/train.jsonl --val-input data/raw/tinystories-1k/val.jsonl --output data/tinystories-1k-eos --block-size 128 --vocab-size 512 --tokenizer-training-bytes 65536 --add-eos
.venv/Scripts/python scripts/run_comparison.py --data-dir data/tinystories-1k-eos --output-dir checkpoints/comparison-1k-200 --steps 200 --seeds 42 43 44 --device cuda --timeout-seconds 180
.venv/Scripts/python scripts/report_experiment.py --comparison-dir checkpoints/comparison-1k-200 --output-dir reports/reproduced-1k --window-steps 20
```

For report-only reproduction from the archived evidence, no GPU, corpus, or checkpoints are needed:

```powershell
.venv/Scripts/python scripts/report_experiment.py --comparison-dir reports/tinystories-1k-200 --output-dir reports/recomputed-1k --window-steps 20
```

Report paths will differ in the regenerated output, but numeric results and log fingerprints should agree. The manifest records the pre-commit HEAD plus `git_dirty: true`; Python source hashes identify the implementation used at launch. GPU determinism is not enforced, so fresh training may differ numerically and in timing. Original configs, logs, summaries, tokenizer and checkpoints are under `checkpoints/comparison-1k-200/`.

## Verification and next decision

97 tests passed, including early/late endpoint arithmetic, pairing, missing-evaluation rejection and token-budget checks. All six logs have steps 0-200; all source fingerprints and final scores match their manifests, and each routing window contains 40,960 assignments per layer. The public archive retains every logged evaluation and routing count.

Next useful improvement: broader or exhaustive held-out evaluation of the saved final checkpoints, followed by a controlled balancing-coefficient ablation. That would test whether the observed dense/MoE gap survives broader evaluation and separate balancing behavior from training progress. Longer runs or deployment should follow explicit evidence, not an assumption that MoE must outperform dense.
