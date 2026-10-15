"""Local single-process model serving; configure MOELAB_CHECKPOINT at startup."""

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from moelab.inference.generate import generate as sample
from moelab.tokenizer.bpe import BPETokenizer
from moelab.training.loop import load_checkpoint


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.lock = threading.Lock()
    checkpoint = os.getenv("MOELAB_CHECKPOINT")
    if checkpoint:
        torch.set_num_threads(int(os.getenv("MOELAB_THREADS", "2")))
        app.state.model, _ = load_checkpoint(checkpoint, os.getenv("MOELAB_DEVICE", "cpu"))
        app.state.model.eval()
        app.state.tokenizer = BPETokenizer.load(Path(checkpoint).with_name("tokenizer.json"))
        if app.state.tokenizer.vocab_size != app.state.model.cfg.vocab_size:
            raise ValueError("Tokenizer and checkpoint vocabulary mismatch")
    yield


app = FastAPI(title="moelab", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())
REQUESTS = Counter("moelab_requests_total", "Total generate requests")
LATENCY = Histogram("moelab_request_seconds", "End-to-end request latency")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(default=40, ge=1)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "ts": time.time(),
    }


@app.get("/ready")
def ready():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(503, "Set MOELAB_CHECKPOINT to load a trained model")
    return {"status": "ready"}


@app.post("/generate")
def generate(req: GenerateRequest):
    ready()
    model, tokenizer = app.state.model, app.state.tokenizer
    ids = tokenizer.encode(req.prompt)
    if len(ids) > model.cfg.max_seq_len:
        raise HTTPException(422, f"Prompt exceeds {model.cfg.max_seq_len} tokens")
    REQUESTS.inc()
    with LATENCY.time(), app.state.lock:
        x = torch.tensor([ids], device=next(model.parameters()).device)
        result = sample(model, x, req.max_new_tokens, req.temperature, req.top_k, req.top_p)
    return {"text": tokenizer.decode(result[0].tolist()), "generated_tokens": req.max_new_tokens}
