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

The original `<|endoftext|>` separator is consumed while converting whole stories to JSONL; it is not added to the model vocabulary. The next preparation command performs exact deduplication and inserts its documented newline boundary. Keep the raw manifest/card/notice beside your prepared artifacts to retain provenance. Existing output directories, incomplete requested subsets, malformed UTF-8, oversized stories, and unexpected card license metadata fail rather than publish partial output. Alternative revisions must be immutable 40-character lowercase commit IDs.

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

By default, preparation uses the complete 256-byte vocabulary without learned merges. Optional BPE fitting is described below. Documents and their two-newline separators are encoded separately, **not with an EOS token or attention barrier**; training windows may cross document boundaries within a split. The existing `prepare_data.py` command remains available for the single-text demo.

The new output directory contains token files, `tokenizer.json`, and `metadata.json` with source/artifact hashes, token counts, normalization rules, and retained/duplicate/overlap counts. Existing output directories are rejected. Malformed records, empty documents, or insufficient retained tokens fail before publishing the output directory. Dataset downloading is available above. Near-duplicate detection, dedicated EOS handling, and a high-performance tokenizer backend remain future work; the bounded reference BPE path below is implemented.

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

`--vocab-size` is the requested capacity; `metadata.json` and `tokenizer.vocab_size` report the actual decodable vocabulary (`256 + learned merges`). For example, requesting 512 on a tiny sample may produce only 271 entries. Training, serving, and the generation CLI reject a model whose vocabulary does not match the tokenizer's actual size.

Fitting uses at most `--tokenizer-training-bytes` UTF-8 bytes (default 65,536) from the prefix of retained, deduplicated, overlap-free **training** documents. No pair counts or merges cross document boundaries, and separator tokens are not part of the fitting sample. The last sampled document may be cut at a byte boundary; whole retained documents are still encoded afterward. Metadata records requested/actual vocabulary, sample byte/document counts, truncation, and a hash of length-prefixed sample documents. With requested size 256, fitting is skipped and sample counts are zero.

The sample limit bounds BPE fitting input, not total RAM or runtime: Python lists/counters add overhead, each merge scans the sample again, and full documents are still read and encoded. Preparation temporarily spools normalized retained documents to disk, then validates both splits' lengths **after** compression. This remains an educational tokenizer, not a production-speed backend.

Tokenizer JSON now uses format version 2 and records both requested and actual sizes. Legacy files remain loadable and their actual size is derived from their merges. Legacy checkpoints created with inflated output vocabularies cannot safely use that smaller tokenizer: prepare fresh data and train a correctly sized model. Existing valid byte-only runs remain loadable using their original data/tokenizer files. Do not regenerate a run's data before resuming: serialization changes can change the hashes checked by resume. Changing tokenization also changes the meaning of token-level perplexity; do not directly compare it with the earlier byte-only number.

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

This is a local synchronous demo. Generation recomputes a sliding context window; it has no KV cache yet. Prompts longer than the model context are rejected by the API. A tokenizer JSON must accompany its matching checkpoint. Public deployment still needs authentication, request limits, and infrastructure verification.

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
