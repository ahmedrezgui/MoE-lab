# Larger bounded story-training experiment

This experiment compares the existing 1,000-story, 200-step demo with fresh training on 5,000 stories for 2,000 steps. Both use seed 42, the same architecture, coefficient 0.1, 512-entry tokenizer, and 100 development stories. The tokenizer and development-stream fingerprints must match before training starts. The larger preparation fits the same bounded training prefix and produces a byte-identical tokenizer.

This is a combined data-and-training-budget experiment. It cannot tell us how much improvement comes from more data versus more updates. The cosine learning-rate schedule also stretches with the training horizon; warmup remains 10 steps. Evaluation/checkpoint intervals change to 100/500 steps to bound overhead. The final checkpoint is selected in advance, not whichever checkpoint scores best.

## Reproduce

Run from the repository root in the existing CUDA environment. Each output directory must be new; choose alternative paths to preserve previous evidence.

```powershell
.venv/Scripts/python scripts/download_data.py --output data/raw/tinystories-5k --train-documents 5000 --val-documents 100
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/tinystories-5k/train.jsonl --val-input data/raw/tinystories-5k/val.jsonl --output data/tinystories-5k-eos --block-size 128 --vocab-size 512 --tokenizer-training-bytes 65536 --add-eos
.venv/Scripts/python scripts/run_story_experiment.py
```

The runner invokes the existing isolated training/full-validation worker with a 900-second deadline. Timeout or failure records a failed report; partial checkpoints are not reported as a completed experiment. CPU scoring and six-sample export are outside that worker deadline. No Azure resources or paid services are involved.

## Measurement protocol

- Training: 2,048,000 next-token targets (2,000 updates x 2 sequences x 4 accumulation batches x 128 positions), sampled with replacement. This is not one guaranteed traversal of the corpus.
- Data: first 5,000 training and first 100 validation stories from the pinned source revision. Exact normalized duplicates and training/development overlaps are removed. This does not detect semantic or approximate duplicates.
- Full development scoring: all 28,667 targets, including the short final block; context resets every 128 positions. No balancing penalty is included in cross entropy. Both checkpoints are scored on CPU in fp32 for the headline comparison; candidate CUDA scoring is retained separately.
- Generation: the same three prompts, greedy and temperature 0.8, seed 42, 64 new tokens, top-k 40 and top-p 0.95, all on CPU. All six outputs are retained before and after.
- Routing: aggregate assignment percentages for steps 1-100 and 1901-2000. Counts describe expert traffic, not semantic expertise.
- Runtime and memory: synchronized training-loop duration includes periodic evaluation, logging and checkpoint writes; PyTorch allocator peaks are not total laptop GPU memory usage. The idle earlier Gradio process remained resident during training.

The old confirmation dataset is not scored or used to select this candidate. The repeatedly inspected development set is not an untouched test. One training seed and six short prompts cannot establish broad capability or statistical reliability.

## Evidence

`experiment.json` records controls, source/data/checkpoint fingerprints, CPU comparison, full CUDA evaluation, and routing. `metrics.jsonl` contains every training step; `run_summary.json` records runtime/memory/environment. `data-metadata.json` records preparation; `samples-before.json` and `samples-after.json` retain every output. Weights and tokenized data remain local, ignored artifacts.


## Recorded result (2026-10-16)

| Measurement | Before: 1k stories / 200 steps | After: 5k stories / 2,000 steps |
| --- | ---: | ---: |
| Full development CE, CPU fp32 | 3.937743 | 2.198544 |
| Full development perplexity | 51.3027 | 9.0119 |
| Scored targets | 28,667 | 28,667 |
| Training targets processed | 204,800 | 2,048,000 |

CE difference (after minus before): **-1.739198**. This is a development improvement under the fixed scoring protocol, not proof of general story quality. Candidate training-loop duration: **237.11 seconds**; worker including setup/full CUDA evaluation: **242.88 seconds**. Peak tensor allocation: **45.17 MiB**; peak reserved allocator memory: **50.00 MiB**. The test suite ran concurrently during roughly the first 50 seconds, and the old idle UI was resident; timing is not an isolated throughput benchmark.

Candidate checkpoint SHA-256: `a0aa2ab93c9628b75ddd81fc91dcba80b0cf8cb7f39686828628035c15b2b10b`. Its local path is `checkpoints/story-5k-2k/last.pt`. All 5,000 training documents were retained (1,817,852 tokens); no exact development overlaps or internal duplicates were found. Processed training targets are about 1.13 times corpus length, but random block sampling does not guarantee coverage.

| Logged layer ID | Early shares %, experts 0/1/2/3 | Late shares %, experts 0/1/2/3 |
| --- | --- | --- |
| 11 | 24.08 / 25.35 / 25.15 / 25.43 | 24.76 / 26.03 / 25.25 / 23.95 |
| 38 | 25.15 / 24.90 / 24.57 / 25.38 | 23.62 / 24.95 / 25.55 / 25.88 |

Traffic stays approximately balanced in these windows. This does not establish expert specialization.

The fixed outputs have more recognizable word and sentence structure, but remain weak stories. For example, greedy generation after `The little fox` begins `had a big smile on his friends` and later switches to `she`. The stochastic sample begins `went to the park. Jack thought that she couldn't wait to take a hurt her.` The longer run still repeats phrases, loses subject consistency and produces malformed endings. **All six samples hit the 64-token cap; none generated EOS.** The complete before/after files are retained, including all unsuccessful samples.

Try the candidate locally while keeping the original UI available:

```powershell
.venv/Scripts/python scripts/demo_ui.py --checkpoint checkpoints/story-5k-2k/last.pt --port 7861
```

Validation: 145 tests passed; lint, formatting and whitespace checks passed. The next useful experiment would isolate training duration on this fixed 5k corpus, with a predeclared budget and the same sample suite, before expanding architecture or claiming fluent generation.
