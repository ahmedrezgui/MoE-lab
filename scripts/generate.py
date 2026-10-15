"""Generate text from a local checkpoint (temperature zero means greedy)."""

import argparse
from pathlib import Path

import torch

from moelab.inference.generate import generate
from moelab.tokenizer.bpe import BPETokenizer
from moelab.training.loop import load_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/demo/last.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    tokenizer = BPETokenizer.load(Path(args.checkpoint).with_name("tokenizer.json"))
    ids = torch.tensor([tokenizer.encode(args.prompt)], device=args.device)
    result = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(result[0].tolist()))
