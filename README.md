# moelab: a Mixture-of-Experts language model from scratch

An educational decoder-only language model in PyTorch. Train random weights, inspect top-k routing, save/resume a run, and generate text locally. Supports a dense FFN baseline and a sparse MoE variant. This is an architecture laboratory, not an instruction-tuned assistant.

The implementation lives in `src/moelab/`; runnable commands and the project roadmap are below.

## Implemented

- Byte-level BPE with train/encode/decode/save/load; UTF-8 text preparation and uint16 memmap blocks.
- Causal GQA + RoPE, RMSNorm, SwiGLU, tied input/output embeddings.
- Dropless top-k routing, auxiliary balancing loss, per-layer assignment counts.
- AdamW, warmup/cosine decay, accumulation, clipping, CPU fp32 and CUDA bf16/fp16.
- JSONL metrics, validation CE/perplexity, checkpointing with optimizer and RNG state.
- Reference sampling with temperature/top-k/top-p; FastAPI generation, readiness, Prometheus metrics.
- Tests for causality, routing equivalence, gradient flow, overfitting, resume, and API behavior.

KV cache, streaming, quantization, real-corpus experiments, Azure infrastructure, and hosted demos are planned. MLflow has an optional wrapper but is not connected to the current trainer. CI configuration is included; remote execution is not yet verified.

## Windows quickstart

From the project directory (Python 3.11+):

```powershell
python -m venv .venv
# CUDA build used for the local NVIDIA GPU:
.venv/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e ".[dev,serve]"
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/prepare_data.py --input examples/demo.txt --output data/demo
.venv/Scripts/python scripts/train.py --config configs/training/demo.yaml --model configs/model/demo-moe.yaml
.venv/Scripts/python scripts/generate.py --prompt "Once upon a time" --temperature 0.8
```

For CPU-only installation, use the PyTorch `/whl/cpu` index instead. On Linux/macOS use `.venv/bin/python`. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for other hardware. No activation or Make is required.

The included original short story is a plumbing fixture. A 200-step run can show loss reduction but does not establish general language quality. The default byte vocabulary is easy to inspect. To explore BPE merges request `--vocab-size 512`, then set the model vocabulary to the **actual** `vocab_size` reported in the prepared metadata. The request is an upper bound: fitting can stop earlier. After compression each split must still contain more tokens than the context length.

## Download a bounded TinyStories subset

The downloader uses the original [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) train/validation files, pinned to revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md) identifies the data as synthetic stories and declares [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).

```powershell
.venv/Scripts/python scripts/download_data.py --output data/raw/tinystories --train-documents 1000 --val-documents 100
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/tinystories/train.jsonl --val-input data/raw/tinystories/val.jsonl --output data/tinystories --block-size 128
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/laptop-moe.yaml --data-dir data/tinystories --device cuda --output-dir checkpoints/tinystories --max-steps 20
```

The download selects the first N nonempty complete stories in each original split, not a random sample. Defaults limit application reads to 32 MiB per source split and reject source stories larger than 1 MiB. Set `--max-bytes-per-split` and `--max-document-bytes` in bytes to adjust these limits. A separately bounded dataset-card request reads at most 64 KiB plus one detection byte. Requests use a 30-second network timeout; there is no automatic retry or partial-download resume.

Output contains `train.jsonl`, `val.jsonl`, `manifest.json`, `SOURCE_README.md`, and `NOTICE.txt`. The manifest records the pinned source URLs, selection/modification policy, license reference, document counts, bytes read, source-prefix hashes, and output hashes. A source-prefix hash covers only the bytes read, possibly including read-ahead beyond the final selected story; it is not a checksum of the entire remote corpus. Transport buffering/HTTP overhead is outside the application-read count.

The original `<|endoftext|>` separator is consumed while converting whole stories to JSONL; the downloader does not assign it a token ID. Preparation performs exact deduplication and inserts a newline boundary by default, or a dedicated EOS token with `--add-eos`. Keep the raw manifest/card/notice beside your prepared artifacts to retain provenance. Existing output directories, incomplete requested subsets, malformed UTF-8, oversized stories, and unexpected card license metadata fail rather than publish partial output. Alternative revisions must be immutable 40-character lowercase commit IDs.

No new dependency is required. Downloaded stories and generated checkpoints stay ignored by Git. This small prefix is suitable for pipeline development; it is not a representative corpus benchmark or evidence of model quality.

## Prepare separate document splits

For local corpora already divided into training and validation, use JSONL files with one object per document:

```json
{"text": "The complete text of one story."}
{"text": "The complete text of another story."}
```

Use different files for the two splits and enough text in each retained split to exceed the model context length:

```powershell
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/train.jsonl --val-input data/raw/val.jsonl --output data/stories --block-size 128
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/laptop-moe.yaml --data-dir data/stories --device cuda --output-dir checkpoints/stories
```

This path preserves whole documents in their source split, standardizes line endings and outer whitespace, removes exact duplicates, and removes training documents also present in validation. Validation takes precedence. It streams text and stores document hashes in memory; it does not detect near-duplicates or establish the source's license.

By default, preparation uses the complete 256-byte vocabulary without learned merges. Optional BPE fitting is described below. Documents and their two-newline separators are encoded separately, with no attention barrier; opt-in EOS is described below; training windows may cross document boundaries within a split. The existing `prepare_data.py` command remains available for the single-text demo.

The new output directory contains token files, `tokenizer.json`, and `metadata.json` with source/artifact hashes, token counts, normalization rules, and retained/duplicate/overlap counts. Existing output directories are rejected. Malformed records, empty documents, or insufficient retained tokens fail before publishing the output directory. Dataset downloading is available above. Near-duplicate detection and a high-performance tokenizer backend remain future work; the bounded reference BPE path below is implemented.

## Fit a bounded BPE tokenizer

```powershell
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/tinystories/train.jsonl --val-input data/raw/tinystories/val.jsonl --output data/tinystories-bpe --block-size 128 --vocab-size 512 --tokenizer-training-bytes 65536
(Get-Content data/tinystories-bpe/metadata.json | ConvertFrom-Json).vocab_size
Copy-Item configs/model/laptop-moe.yaml configs/model/laptop-bpe.yaml
```

Set `vocab_size` in `configs/model/laptop-bpe.yaml` to the number printed by the metadata command, then train with that config and a fresh run directory:

```powershell
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/laptop-bpe.yaml --data-dir data/tinystories-bpe --device cuda --output-dir checkpoints/tinystories-bpe --max-steps 20
```

`--vocab-size` is the requested capacity; `metadata.json` and `tokenizer.vocab_size` report the actual decodable vocabulary (`256 + learned merges + 1 if EOS is enabled`, otherwise `256 + learned merges`). For example, requesting 512 on a tiny sample may produce only 271 entries. Training, serving, and the generation CLI reject a model whose vocabulary does not match the tokenizer's actual size.

Fitting uses at most `--tokenizer-training-bytes` UTF-8 bytes (default 65,536) from the prefix of retained, deduplicated, overlap-free **training** documents. No pair counts or merges cross document boundaries, and separator tokens are not part of the fitting sample. The last sampled document may be cut at a byte boundary; whole retained documents are still encoded afterward. Metadata records requested/actual vocabulary, sample byte/document counts, truncation, and a hash of length-prefixed sample documents. With requested size 256, fitting is skipped and sample counts are zero.

The sample limit bounds BPE fitting input, not total RAM or runtime: Python lists/counters add overhead, each merge scans the sample again, and full documents are still read and encoded. Preparation temporarily spools normalized retained documents to disk, then validates both splits' lengths **after** compression. This remains an educational tokenizer, not a production-speed backend.

Tokenizer JSON now uses format version 3 and records requested/actual sizes and EOS configuration. Formats 1 and 2 load without EOS. Legacy files remain loadable and their actual size is derived from their merges. Legacy checkpoints created with inflated output vocabularies cannot safely use that smaller tokenizer: prepare fresh data and train a correctly sized model. Existing valid byte-only runs remain loadable using their original data/tokenizer files. Do not regenerate a run's data before resuming: serialization changes can change the hashes checked by resume. Changing tokenization also changes the meaning of token-level perplexity; do not directly compare it with the earlier byte-only number.

## Optional end-of-sequence token

EOS (end of sequence) gives the model a token it can learn to predict at the end of a document:

```powershell
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/tinystories/train.jsonl --val-input data/raw/tinystories/val.jsonl --output data/tinystories-eos --block-size 128 --vocab-size 512 --tokenizer-training-bytes 65536 --add-eos
(Get-Content data/tinystories-eos/metadata.json | ConvertFrom-Json).vocab_size
```

Create a fresh model configuration with that actual vocabulary size and train in a fresh run directory, using `--data-dir data/tinystories-eos`. The requested capacity includes EOS and must be at least 257; 257 gives byte-only text plus EOS. EOS has ID `256 + number of learned merges`. Metadata records this ID. Preparation appends exactly one EOS after each retained document, replacing the default two-newline separator. EOS never participates in BPE fitting.

EOS is an ordinary prediction target during training. It does not reset positions or attention, and packed windows can still cross documents. There is no loss masking at boundaries: predicting the next document after EOS is also part of training. Compare validation loss only with the same tokenizer and boundary policy.

The CLI and API stop on a newly generated EOS or the token limit. Prompts do not receive EOS automatically; literal `<|endoftext|>` text stays ordinary text. Decoding hides special tokens by default; `decode(ids, skip_special_tokens=False)` displays EOS as `<|endoftext|>` for inspection. A short training run need not learn when to stop.

The Python sampler returns EOS in its token tensor. For batches, finished rows receive repeated EOS padding until every row finishes or the limit is reached; slice each continuation at its first EOS to recover its length. Prompt EOS tokens do not stop generation. No separate padding token is introduced. Existing tokenizers without EOS keep fixed-length generation. Enabling EOS requires freshly prepared data and a new training run, not resuming an old checkpoint with a replaced tokenizer.

## Your NVIDIA laptop

Detected hardware: **RTX 3050 Ti Laptop GPU, 4 GB VRAM**. The conservative laptop preset uses two layers, width 128, four experts, top-2 routing, context 128, and microbatch 2 with four-step accumulation.

```powershell
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/laptop-moe.yaml --device cuda --output-dir checkpoints/laptop
.venv/Scripts/python scripts/generate.py --checkpoint checkpoints/laptop/last.pt --device cuda --prompt "Once upon a time"
```

Use `--max-steps 20` for a bounded smoke run. Resume toward the original configured horizon with the same config/data:

```powershell
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/laptop-moe.yaml --device cuda --output-dir checkpoints/laptop --resume checkpoints/laptop/last.pt
```

A fresh run requires a fresh output directory. Each run saves `last.pt`, `best.pt`, `tokenizer.json`, and `metrics.jsonl`. Training loss includes the auxiliary term; `val_ce` and `val_perplexity` do not. Perplexity is comparable only with the same tokenizer and evaluation protocol. The tiny validation split is diagnostic, not a publishable benchmark.

`last.pt` records the latest scheduled checkpoint (and the end of each run) for resuming training. `best.pt` saves a full checkpoint whenever a **trained** model achieves strictly lower validation cross-entropy, even between regular checkpoint intervals. Ties keep the earlier best; the step-zero evaluation is diagnostic only. Evaluation rows and checkpoints record `best_val_ce` and `best_step`. Non-finite validation CE stops training without replacing either checkpoint.

Resume in the same output directory to retain the existing `best.pt` alongside its selection history. If you resume into a different directory, the historical best file stays in the original directory until a new improvement produces a new `best.pt`. Older checkpoints without selection metadata remain supported: selection starts afresh at the first evaluation after resuming; past best weights cannot be reconstructed. Existing demo artifacts will not gain a best checkpoint retroactively.

Generate from the best model after a new training run:

```powershell
.venv/Scripts/python scripts/generate.py --checkpoint checkpoints/laptop/best.pt --device cuda --prompt "Once upon a time"
```

## Serve a checkpoint

```powershell
$env:MOELAB_CHECKPOINT = "checkpoints/laptop/last.pt"
# Optional: $env:MOELAB_DEVICE = "cuda" (CPU is the serving default)
.venv/Scripts/python -m uvicorn moelab.serving.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try `/generate`. `/health` checks the process; `/ready` returns 503 until a checkpoint is loaded. Example PowerShell request:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -ContentType "application/json" -Body '{"prompt":"Once upon a time","max_new_tokens":60,"temperature":0.8}'
```

`/generate` returns `text`, `generated_tokens`, and `finish_reason` (`"eos"` or `"length"`). The count excludes the prompt and includes a generated EOS even though it is hidden in text.

This is a local synchronous demo. Generation recomputes a sliding context window; it has no KV cache yet. Prompts longer than the model context are rejected by the API. A tokenizer JSON must accompany its matching checkpoint. Public deployment still needs authentication, request limits, and infrastructure verification.

## Controlled dense-versus-MoE budgets

Use `configs/model/compare-dense.yaml` and `configs/model/compare-moe.yaml` with the same `configs/training/laptop.yaml`. They share vocabulary, width 128, two layers, attention heads, context 128, and zero dropout. The comparison matches estimated **active parameters per token**, including the router, within 0.1%.

| Model (vocabulary 256) | FFN hidden width | Total parameters | Active estimate |
| --- | ---: | ---: | ---: |
| Dense | 683 | 656,256 | 656,256 |
| MoE, four experts / top-2 | 341 per expert | 1,180,288 | 656,512 |

The signed MoE-minus-dense active gap is 256 weights, or 0.039% of dense. One SwiGLU has `3 * d_model * ffn_hidden_dim` weights. Top-2 activates two such experts; routing adds `d_model * n_experts` weights per layer. The ideal dense hidden width is `2 * 341 + 4 / 3 = 683.333...`; 683 is the nearest integer. Tied embeddings are counted once; all attention and normalization weights are included. RoPE buffers are not trainable parameters.

```powershell
.venv/Scripts/python scripts/compare_budgets.py
```

The command prints a JSON breakdown without allocating model weights, rejects mismatched backbones, and exits unsuccessfully when the active gap exceeds 0.1%. Use `--dense PATH --moe PATH` for edited configurations. `ffn_hidden_dim` controls dense width or each expert's width. Omitting it preserves the previous `int(d_model * (8 / 3))` default, including old checkpoints.

Before training, set `vocab_size` in **both** comparison configs to the actual value in your prepared `metadata.json`, then rerun the budget command. Both models must use exactly the same prepared token files and tokenizer, including EOS policy. For example, with the EOS corpus prepared above:

```powershell
# First update both vocab_size values from data/tinystories-eos/metadata.json.
.venv/Scripts/python scripts/compare_budgets.py
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/compare-dense.yaml --data-dir data/tinystories-eos --device cuda --output-dir checkpoints/compare-dense --max-steps 20
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/compare-moe.yaml --data-dir data/tinystories-eos --device cuda --output-dir checkpoints/compare-moe --max-steps 20
```

These are bounded pipeline runs, not completed quality experiments. Both use the same seed, optimizer settings, precision, evaluation schedule, and 1,024 training tokens per optimizer step (`2 * 128 * 4`). The shared training horizon remains 200 steps, even when stopping at 20; the learning-rate schedule is therefore unchanged. The trainer's independent NumPy sampling generator supplies identical training windows for equal seeds and step counts; evaluation uses its own fixed seed. Different architectures do not have identical initial weights merely because the seed matches.

This is not a match on total capacity, FLOPs, wall-clock time, or peak GPU memory. MoE stores more weights and optimizer state, and routing/dispatch adds overhead. The active count includes shared weights and selected expert matrices, not the number of distinct weights touched by an entire batch. Compare validation CE (which excludes auxiliary balancing loss), not raw training loss (which includes it for MoE). Report final-step results at equal token budgets; label best-checkpoint results separately because selected steps can differ. Repeat paired seeds before drawing conclusions. The bounded runner below records timing, memory, and validation; no general performance advantage is claimed.

## Run a bounded local GPU comparison

```powershell
.venv/Scripts/python scripts/run_comparison.py --data-dir data/tinystories-eos --output-dir checkpoints/comparison --steps 20 --seeds 42 43 --device cuda --timeout-seconds 180
```

The runner automatically sets both vocabularies from the prepared tokenizer, checks the 0.1% active-budget tolerance and shared backbone, and uses the same training configuration for both models. It launches each run sequentially in a fresh Python process, alternating model order by seed. Defaults use the comparison model presets and `configs/training/laptop.yaml`; override them with `--dense`, `--moe`, and `--config`. CPU checks require a training config with `precision: fp32` and `--device cpu`. Requested CUDA failures are surfaced, not silently changed to CPU.

The output directory must be new. It contains `comparison.json` and one directory per model/seed, each with resolved YAML configs, `console.log`, `metrics.jsonl`, `run_summary.json`, and checkpoints. The comparison manifest records source hashes, Git HEAD/dirty state, corpus fingerprints, actual model budgets, software/GPU details, execution order and results. Partial completed results are preserved with `status: failed` if a worker fails or times out; inspect its console log and use a fresh output directory to retry. The timeout is per process, including startup, not per training step. Defaults bound the experiment to four processes, each at most 180 seconds.

`run_summary.json` is also produced by ordinary training. It records synchronized training-loop seconds and tokens/s including evaluation, logging and checkpoint writes, excluding setup and initial validation. The runner additionally records whole-process seconds. Peak allocated/reserved MiB describe PyTorch's GPU allocator during the loop; they exclude total CUDA/driver usage and other applications. CPU memory fields are null. Summary files describe the last successful session; on resume, session token counts/time cover only newly executed steps, while best-checkpoint history can refer to earlier steps. Per-step legacy timing fields in `metrics.jsonl` have different boundaries; use summary fields for comparisons.

A completed [local GPU smoke report](reports/local-gpu-smoke.md) and its [full JSON record](reports/local-gpu-smoke.json) cover two paired seeds, 20 steps each, on 8 training and 4 validation TinyStories documents. Mean final CE was 3.4291 dense versus 3.4733 MoE; mean loop time was 1.56 versus 2.92 seconds, with peak allocated memory about 33.63 versus 43.82 MiB. These tiny runs validate the measurement pipeline and do not establish general language quality or an architecture ranking.

## Inspect expert utilization

Generate reports from existing MoE training logs without loading a checkpoint or retraining:

```powershell
.venv/Scripts/python scripts/report_routing.py --metrics checkpoints/comparison/moe-seed-42/metrics.jsonl checkpoints/comparison/moe-seed-43/metrics.jsonl --output-dir reports/my-routing
```

The output directory must be new. `routing.md` contains per-layer/per-expert assignment percentages and concentration summaries; `routing.json` also retains every observed step's counts and percentages. Use `--start-step 11 --end-step 20` to examine a later window. Bounds are inclusive; the report records both requested bounds and actual observed steps. Source log hashes make the input traceable. Dense logs, non-finite/negative/fractional counts, changing layer layouts, inconsistent layer totals, and missing steps inside an observed window are rejected.

Assignment percentages are calculated by summing counts over the selected steps and dividing by each layer's total assignments. They sum to 100% per layer. With top-2 routing, one token contributes two assignments, so 25% assignment share is not 25% token coverage. Counts describe hard selection, not gate weights, expert quality, or GPU time. Layer IDs such as `11` and `38` come from the trainer's module enumeration; they are not decoder block indices.

The report shows each expert's number of zero-assignment steps, aggregate busiest-expert share relative to uniform, and the highest per-step ratio. With four experts, uniform is 25%; a max/uniform ratio of 1.6 means the busiest expert received 40%. Whole-run balance can hide temporary concentration, so inspect per-step data or windows before interpreting it. This implementation selects distinct experts per token; with four experts and top-2, an expert's maximum assignment share is 50% and the maximum ratio is 2.

The [recorded smoke routing report](reports/routing-gpu-smoke/routing.md) covers both completed GPU seeds. All experts received assignments across each full run, but several experts in logged layer `38` had individual steps with zero assignments. For seed 42, expert 2 had three such steps despite receiving 29.38% of that layer's total assignments. This illustrates why an aggregate alone is insufficient. These short runs do not establish expert specialization or persistent routing collapse.

## Larger bounded experiment and early/late routing

The next recorded experiment uses 1,000 training and 100 validation TinyStories documents, a shared 512-entry BPE vocabulary including EOS, and 200 steps for each model over seeds 42, 43 and 44. Reproduce it with fresh output paths:

```powershell
.venv/Scripts/python scripts/download_data.py --output data/raw/tinystories-1k --train-documents 1000 --val-documents 100
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/tinystories-1k/train.jsonl --val-input data/raw/tinystories-1k/val.jsonl --output data/tinystories-1k-eos --block-size 128 --vocab-size 512 --tokenizer-training-bytes 65536 --add-eos
.venv/Scripts/python scripts/run_comparison.py --data-dir data/tinystories-1k-eos --output-dir checkpoints/comparison-1k-200 --steps 200 --seeds 42 43 44 --device cuda --timeout-seconds 180
.venv/Scripts/python scripts/report_experiment.py --comparison-dir checkpoints/comparison-1k-200 --output-dir reports/tinystories-1k-200 --window-steps 20
```

The report pairs validation CE at step 20 and step 200 with routing over steps 1-20 and 181-200, respectively. Windows are fixed before inspecting results and must be nonoverlapping. It preserves every recorded validation evaluation and both windows' per-step routing in `analysis.json`, with readable tables in `analysis.md`. Missing step records, incomplete comparisons, unequal paired token budgets, or a missing early-window endpoint evaluation fail explicitly; validation is never interpolated. The default reporter expects the four-expert experiment presets for its displayed expert ordering.

See the [experiment protocol and interpretation](reports/tinystories-1k-200/README.md) and [validation/routing tables](reports/tinystories-1k-200/analysis.md). This is still a small experiment: each run processes 204,800 targets sampled with replacement, and validation measures the same 1,024 sampled targets rather than all 28,668 held-out tokens. Three seeds provide descriptive variability, not a general architecture conclusion. The tokenizer differs from the earlier smoke fixture, so their token-level CE/perplexity values are not directly comparable.

## Tests and experiments

```powershell
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m pytest
```

The dense and MoE `tiny` configs share a backbone but **do not have matched active parameters**: top-2 runs two full-sized FFNs per token. `num_params(active_only=True)` reports an architectural estimate. A fair comparison needs an explicit parameter/compute budget and measured throughput.

Suggested portfolio sequence: reproduce the local demo, train on a documented corpus, plot expert utilization with/without balancing, compare dense/MoE under a declared budget, measure KV-cache speedups, then deploy and publish the evidence. Do not claim pretrained-level language quality or MoE speed advantages from the smoke test.

## Recorded runs and cloud assessment

A 200-step toy-corpus run completed on the local RTX 3050 Ti. Validation cross-entropy fell from 5.55 initially to 2.22 at step 100, then rose to 2.57 at step 200 as the model overfit. These are pipeline checks, not general language-quality benchmarks. Training currently runs locally; no Azure deployment has been made.
