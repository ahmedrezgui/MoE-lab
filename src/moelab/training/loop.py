"""Single-device AdamW training with reproducible resume and JSONL metrics."""

import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from moelab.config import ModelConfig, TrainingConfig
from moelab.data.dataset import BlockDataset
from moelab.evals.perplexity import evaluate
from moelab.model.moe import MoELayer
from moelab.model.transformer import Transformer
from moelab.tokenizer.bpe import BPETokenizer


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig.model_validate(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(payload, path):
    """Replace a checkpoint only after its new contents have been written."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    max_steps=None,
    *,
    data_dir="data/demo",
    output_dir="checkpoints/demo",
    device="cpu",
    resume=None,
):
    c = train_cfg
    stop = c.max_steps if max_steps is None else max_steps
    if not 1 <= stop <= c.max_steps:
        raise ValueError("max_steps override must be between 1 and the configured training horizon")
    device = torch.device(device)
    if device.type != "cuda" and c.precision != "fp32":
        raise ValueError("Use precision: fp32 for CPU training")
    if device.type == "cuda" and c.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; choose fp16 or fp32")
    torch.manual_seed(c.seed)
    rng = np.random.default_rng(c.seed)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    tokenizer = BPETokenizer.load(data_dir / "tokenizer.json")
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError("Tokenizer and model vocabulary sizes differ")
    datasets = {
        s: BlockDataset(data_dir / f"ids_{s}.bin", model_cfg.max_seq_len) for s in ("train", "val")
    }
    fingerprints = {
        name: file_hash(data_dir / name)
        for name in ("ids_train.bin", "ids_val.bin", "tokenizer.json")
    }
    for dataset in datasets.values():
        if int(dataset.ids.max()) >= model_cfg.vocab_size:
            raise ValueError("Token ID outside model vocabulary")
    model = Transformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and c.precision == "fp16")
    step = 0
    best_val_ce = math.inf
    best_step = None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["model_config"] != model_cfg.model_dump():
            raise ValueError("Resume model configuration mismatch")
        if saved["training_config"] != c.model_dump() or saved["data_hashes"] != fingerprints:
            raise ValueError("Resume requires the same training config and tokenized corpus")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step = saved["step"]
        best_val_ce = saved.get("best_val_ce", math.inf)
        best_step = saved.get("best_step")
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if step >= stop:
        raise ValueError("Checkpoint has already reached the requested step")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if not resume and metrics_path.exists():
        raise ValueError("Output already contains a run; use a new output directory or --resume")
    tokenizer.save(output_dir / "tokenizer.json")
    if not resume:
        initial = {
            "step": 0,
            **evaluate(model, datasets["val"], batches=4, batch_size=c.batch_size),
        }
        metrics_path.write_text(json.dumps(initial) + "\n", encoding="utf-8")
        print(json.dumps(initial))
    print(
        json.dumps(
            {
                "total_params": model.num_params(),
                "active_params_estimate": model.num_params(True),
                "device": str(device),
            }
        )
    )
    start = time.perf_counter()
    start_step = step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < stop:
        if step < c.warmup_steps:
            factor = (step + 1) / c.warmup_steps
        else:
            progress = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = c.lr * factor
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        routing_counts = {
            str(i): torch.zeros_like(m.token_counts)
            for i, m in enumerate(model.modules())
            if isinstance(m, MoELayer)
        }
        for _ in range(c.grad_accum_steps):
            x, y = datasets["train"].get_batch(c.batch_size, rng)
            context = (
                torch.autocast(
                    "cuda",
                    dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[c.precision],
                )
                if c.precision != "fp32"
                else nullcontext()
            )
            with context:
                _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            train_loss += loss.item() / c.grad_accum_steps
            scaler.scale(loss / c.grad_accum_steps).backward()
            for i, module in enumerate(model.modules()):
                if isinstance(module, MoELayer):
                    routing_counts[str(i)] += module.token_counts
        scaler.unscale_(optimizer)
        if c.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), c.grad_clip, error_if_nonfinite=not scaler.is_enabled()
            )
        scaler.step(optimizer)
        scaler.update()
        step += 1
        row = {
            "step": step,
            "train_loss": train_loss,
            "lr": c.lr * factor,
            "elapsed_seconds": time.perf_counter() - start,
            "expert_counts": {k: v.tolist() for k, v in routing_counts.items()},
        }
        improved = False
        if step % c.eval_every == 0 or step == stop:
            row.update(evaluate(model, datasets["val"], batches=4, batch_size=c.batch_size))
            if not math.isfinite(row["val_ce"]):
                raise RuntimeError("Non-finite validation cross entropy")
            improved = row["val_ce"] < best_val_ce
            if improved:
                best_val_ce = row["val_ce"]
                best_step = step
            row.update(best_val_ce=best_val_ce, best_step=best_step)
            row["training_tokens"] = (
                step * c.batch_size * c.grad_accum_steps * model_cfg.max_seq_len
            )
            row["session_tokens_per_second"] = (
                (step - start_step)
                * c.batch_size
                * c.grad_accum_steps
                * model_cfg.max_seq_len
                / (time.perf_counter() - start)
            )
            if device.type == "cuda":
                row["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            print(json.dumps(row))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        save_last = step % c.checkpoint_every == 0 or step == stop
        if save_last or improved:
            payload = {
                "model_config": model_cfg.model_dump(),
                "training_config": c.model_dump(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val_ce": best_val_ce,
                "best_step": best_step,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "data_hashes": fingerprints,
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            if improved:
                save_checkpoint(payload, output_dir / "best.pt")
            if save_last:
                save_checkpoint(payload, output_dir / "last.pt")
    return model
