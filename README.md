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

The included original short story is a plumbing fixture. A 200-step run can show loss reduction but does not establish general language quality. The default byte vocabulary is easy to inspect. To explore BPE merges use `--vocab-size 512` and set the model vocabulary to 512; after compression each split must still contain more tokens than the context length.

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
