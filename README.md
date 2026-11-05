# MoE Lab

One from-scratch Mixture of Experts language model, trained locally and served in
an interactive React research dashboard. This is a story-continuation model, not
an instruction-tuned chat assistant.

## The two configuration files

- **`configs/moe.yaml`**: model architecture. 1.21M parameters, two decoder blocks,
  four SwiGLU experts per block, top-2 routing, 512-token BPE vocabulary, 128-token context.
- **`configs/training.yaml`**: training settings. **20,000 updates**, learning rate
  0.001, BF16, batch size 2, four gradient-accumulation batches, seed 42.

There are no separate demo, laptop, portfolio, or comparison model presets.
The React app and CLI demos use **`checkpoints/moe/last.pt`** and its adjacent
`tokenizer.json`. Historical checkpoints are backups, not additional active models.

## Train

From the repository root in the existing CUDA environment:

```powershell
.venv/Scripts/python scripts/train.py --restart
```

This uses the two configs above and `data/tinystories-5k-eos`. It trains fresh
weights for the configured horizon; it does not extend the old optimizer schedule.
`--restart` preserves the previous canonical directory under `checkpoints/archive/`
before beginning a new run. Stop the API before training; restart it afterward.
No training starts just from editing the configuration.

To change the training length, edit `max_steps` in `configs/training.yaml` and run
the same command. To continue an interrupted run with **unchanged** config and data:

```powershell
.venv/Scripts/python scripts/train.py --resume checkpoints/moe/last.pt
```

Training writes `last.pt`, `best.pt`, `metrics.jsonl`, and `run_summary.json` into
`checkpoints/moe`. The app deliberately serves `last.pt`; `best.pt` is a recovery/
evaluation artifact, not a separate model architecture. Development loss is checked
every 250 updates. More updates can overfit; inspect validation loss and samples.

## Serve in the React dashboard

```powershell
.venv/Scripts/python scripts/serve.py
```

This launcher selects the canonical checkpoint, overriding any old
`MOELAB_CHECKPOINT` shell setting. CUDA is used automatically when available;
`MOELAB_DEVICE=cpu` forces CPU. The API listens on `127.0.0.1:8000`.

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. The UI offers story generation, real prompt-token
routing, expert utilization, architecture inspection, and historical experiment
results. Restart the API and refresh the browser after training to load new weights.
The API keeps weights in memory; it does not reload automatically.

## CLI demos

```powershell
.venv/Scripts/python scripts/generate.py --prompt "The little fox" --device cuda
.venv/Scripts/python scripts/demo_samples.py --output reports/my-samples.json
```

Both default to the same canonical checkpoint. The sample exporter has explicit
seeds and refuses to overwrite an existing output file.

## Fresh environment and data

Python 3.11+ and Node.js are required. Install a PyTorch build appropriate to your
hardware first, then install the Python package:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,serve]"
.venv/Scripts/python scripts/download_data.py --output data/raw/tinystories-5k --train-documents 5000 --val-documents 100
.venv/Scripts/python scripts/prepare_documents.py --train-input data/raw/tinystories-5k/train.jsonl --val-input data/raw/tinystories-5k/val.jsonl --output data/tinystories-5k-eos --block-size 128 --vocab-size 512 --tokenizer-training-bytes 65536 --add-eos
```

Use the existing prepared data if already present. Preparation requires fresh
output directories. CPU training requires `precision: fp32` in the training config
and `--device cpu`. No checkpoint or corpus is bundled in Git.

## Validation

```powershell
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
cd frontend
npm run build
npm run lint
```

## Research evidence

`reports/` retains measured experiments and their original protocols and source
hashes. These archives may mention retired preset names or UI commands; use this
README for the current workflow. The dashboard's Experiments page displays the current model's training/validation
loss curves (refreshed every five seconds), followed by the archived dense/MoE comparison.

Historical comparison/evaluation utilities remain available for analyzing evidence;
dense comparison commands require an explicit external dense config. Test-only
architectures are constructed in test fixtures, not user-facing presets.

See [MODEL_CARD.md](MODEL_CARD.md) for limitations and historical evidence.
Training, serving, and inference remain local. No Gradio, cloud service, or hosted
LLM is required. Generated-token routing, KV caching, and streaming are not implemented.

## Terminal progress and essential artifacts

Training prints readable progress every 25 iterations, at evaluation, and at the
first/final update: completed/total iterations, remaining iterations, training
loss, validation CE when measured, speed, and estimated minutes remaining.
Progress is flushed to the terminal; no console log file is created. Because blocks
are randomly sampled, updates are iterations rather than completed epochs.
`metrics.jsonl` remains the compact numerical record needed for loss visualizations.
Training loss includes the balancing objective; validation CE excludes it.

The active local directories are:

- `checkpoints/moe/`: weights, tokenizer, metrics, and run summary. Training also
  writes `best.pt` for recovery/evaluation; `last.pt` remains the served checkpoint.
- `data/tinystories-5k-eos/`: training/validation tokens, tokenizer, and metadata.
- `data/raw/tinystories-5k/`: source stories and download manifest for re-preparation.

Old checkpoint/data artifacts were moved outside the repository to a sibling
`moe-lab-retired-artifacts-*` backup. Historical reports and samples remain in
`reports/`; rerunning their original protocols requires their original artifacts.

`serving/api.py` owns model startup, health, and generation. `serving/workspace.py`
owns the dashboard's metadata, routing and chart endpoints. They share one model
and lock; these are code modules, not separate models. The seeded CLI sample engine
is now `inference/demo.py`, outside serving.
