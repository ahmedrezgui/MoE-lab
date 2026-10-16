"""Resident checkpoint and request-local sampling for the interactive demo."""

import math
import threading
import time
from pathlib import Path

import torch

from moelab.comparison import fingerprint
from moelab.inference.generate import generate
from moelab.tokenizer.bpe import BPETokenizer
from moelab.training.loop import load_checkpoint

DEFAULT_CHECKPOINT = "checkpoints/balance-ablation-1k/coef-0.1-seed-42/last.pt"
EXAMPLE_PROMPTS = ["Once upon a time", "The little fox", "One day, a girl"]


class DemoModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
        path = Path(checkpoint)
        if not path.is_file():
            raise ValueError(
                "Checkpoint not found. Pass --checkpoint PATH; see MODEL_CARD.md for reproduction."
            )
        tokenizer_path = path.with_name("tokenizer.json")
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.model, saved = load_checkpoint(path, device)
        if self.tokenizer.vocab_size != self.model.cfg.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match the checkpoint")
        expected = saved.get("data_hashes", {}).get("tokenizer.json")
        if expected and fingerprint(tokenizer_path) != expected:
            raise ValueError("Tokenizer fingerprint does not match the training checkpoint")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.lock = threading.Lock()
        self.info = {
            "checkpoint": path.as_posix(),
            "checkpoint_sha256": fingerprint(path),
            "tokenizer_sha256": fingerprint(tokenizer_path),
            "step": saved.get("step"),
            "training_seed": saved.get("training_config", {}).get("seed"),
            "device": str(self.device),
            "torch": str(torch.__version__),
            "model_config": self.model.cfg.model_dump(),
            "total_parameters": self.model.num_params(),
            "active_parameters_estimate": self.model.num_params(True),
        }
        del saved

    def complete(self, prompt, max_new_tokens=64, temperature=0.8, top_k=40, top_p=0.95, seed=42):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
            raise ValueError("Enter a nonempty prompt of at most 4,096 characters.")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 256:
            raise ValueError("New-token limit must be an integer from 1 to 256.")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if type(top_k) is not int or not 0 <= top_k <= self.tokenizer.vocab_size:
            raise ValueError("Top-k must be an integer within the vocabulary; 0 disables it.")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("Top-p must be greater than 0 and at most 1.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer between 0 and 4294967295.")
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.model.cfg.max_seq_len:
            raise ValueError(
                f"Prompt is {len(ids)} tokens; this model accepts at most "
                f"{self.model.cfg.max_seq_len}. Shorten the prompt."
            )
        with self.lock:
            rng = torch.Generator(device=self.device).manual_seed(seed)
            x = torch.tensor([ids], device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            result = generate(
                self.model,
                x,
                max_new_tokens,
                temperature,
                top_k or None,
                top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                generator=rng,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - start
            tokens = result[0].tolist()
        new = tokens[len(ids) :]
        return {
            "text": self.tokenizer.decode(tokens),
            "continuation": self.tokenizer.decode(new),
            "prompt_tokens": len(ids),
            "generated_tokens": len(new),
            "finish_reason": "eos" if new[-1] == self.tokenizer.eos_token_id else "length",
            "generation_seconds": seconds,
            "device": str(self.device),
            "settings": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
            },
        }
