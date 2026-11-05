"""Read-only workspace metadata, archived experiments and prompt routing inspection."""

import json
import os
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


def loaded(request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(503, "Set MOELAB_CHECKPOINT and restart the API to load a model.")
    return model


@router.get("/model")
def model_info(request: Request):
    model = loaded(request)
    return {
        "name": Path(os.getenv("MOELAB_CHECKPOINT", "checkpoint")).parent.name,
        "config": model.cfg.model_dump(),
        "total_parameters": model.num_params(),
        "active_parameters_estimate": model.num_params(True),
        "device": str(next(model.parameters()).device),
    }


class InspectRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32768)
    allow_long_text: bool = False


@router.post("/inspect")
def inspect_prompt(body: InspectRequest, request: Request):
    model = loaded(request)
    tokenizer = request.app.state.tokenizer
    ids = tokenizer.encode(body.prompt)
    limit = 2048 if body.allow_long_text else model.cfg.max_seq_len
    if not body.prompt.strip() or not ids or len(ids) > limit:
        raise HTTPException(422, f"Enter 1–{model.cfg.max_seq_len} prompt tokens.")
    layers = {}
    handles = []
    tail_only = False

    def capture(index):
        def hook(module, args, logits):
            probs = logits.detach().float().softmax(-1)
            if tail_only:
                probs = probs[-1:]
            weights, selected = probs.topk(model.cfg.moe.top_k, dim=-1)
            if model.cfg.moe.top_k > 1:
                weights = weights / weights.sum(-1, keepdim=True)
            layer = layers.setdefault(
                index,
                {
                    "layer": index,
                    "probabilities": [],
                    "selected": [],
                    "weights": [],
                    "counts": [0] * model.cfg.moe.n_experts,
                },
            )
            layer["probabilities"].extend(probs.cpu().tolist())
            layer["selected"].extend(selected.cpu().tolist())
            layer["weights"].extend(weights.cpu().tolist())
            counts = torch.bincount(selected.flatten(), minlength=model.cfg.moe.n_experts)
            layer["counts"] = [
                a + b for a, b in zip(layer["counts"], counts.cpu().tolist(), strict=True)
            ]

        return hook

    with request.app.state.lock, torch.inference_mode():
        try:
            if model.cfg.moe:
                for index, block in enumerate(model.blocks):
                    handles.append(block.ffn.router.register_forward_hook(capture(index)))
            x = torch.tensor([ids], device=next(model.parameters()).device)
            context = model.cfg.max_seq_len
            model(x[:, :context])
            # Every later token is inspected with its preceding sliding context.
            # Retain only its own routing row, never duplicate preceding tokens.
            tail_only = True
            if model.cfg.moe:
                for end in range(context + 1, len(ids) + 1):
                    model(x[:, end - context : end])
        finally:
            for handle in handles:
                handle.remove()
    return {
        "scope": "Re-inspection of tokenized text with causal sliding context; "
        "not telemetry captured during generation.",
        "text": body.prompt,
        "tokens": [{"id": token, "text": tokenizer.decode([token])} for token in ids],
        "layers": list(layers.values()),
    }


@router.get("/experiments")
def experiments():
    root = Path(os.getenv("MOELAB_REPORTS_DIR", "reports"))
    path = root / "tinystories-1k-200" / "analysis.json"
    if not path.is_file():
        return {"source": None, "runs": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "source": "reports/tinystories-1k-200/analysis.json",
            "protocol": "1,000 training stories · 200 updates · 1,024 sampled validation targets",
            "runs": [
                {
                    key: run[key]
                    for key in (
                        "model",
                        "seed",
                        "validation_curve",
                        "final_val_ce",
                        "loop_seconds",
                        "loop_tokens_per_second",
                        "peak_allocated_mib",
                        "session_training_tokens",
                    )
                }
                for run in data["runs"]
            ],
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(500, "Archived experiment report could not be read.") from error


@router.get("/training")
def training_history():
    """Read canonical metrics, tolerating an incomplete trailing write during training."""
    root = Path(os.getenv("MOELAB_TRAINING_DIR", "checkpoints/moe"))
    path = root / "metrics.jsonl"
    if not path.is_file():
        return {"points": [], "step": 0, "total_steps": None, "source": str(path)}
    points = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "step" not in row:
                continue
            points.append(
                {
                    key: row[key]
                    for key in ("step", "train_loss", "val_ce", "total_steps")
                    if key in row
                }
            )
        total = points[-1].get("total_steps") if points else None
        summary = root / "run_summary.json"
        if total is None and summary.is_file():
            total = json.loads(summary.read_text(encoding="utf-8"))["training_config"]["max_steps"]
        # Keep validation points and evenly spaced training points to bound chart size.
        stride = max(1, (len(points) + 999) // 1000)
        sampled = [
            p
            for i, p in enumerate(points)
            if i % stride == 0 or "val_ce" in p or i == len(points) - 1
        ]
        return {
            "points": sampled,
            "step": points[-1]["step"] if points else 0,
            "total_steps": total,
            "source": str(path),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(500, "Training metrics could not be read.") from error
