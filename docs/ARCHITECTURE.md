# Architecture walkthrough

Start with `configs/model/demo-moe.yaml` and follow these files in order:

1. `tokenizer/bpe.py`: UTF-8 starts as IDs 0..255. Training repeatedly merges frequent adjacent pairs; encoding applies learned merges in rank order. Unseen Unicode falls back to bytes. No pretrained vocabulary.
2. `data/dataset.py`: sample a contiguous block of T+1 tokens; input is the first T, target is the last T. The model learns next-token prediction, not question-answer pairs.
3. `model/attention.py`: project queries, keys, values; rotate paired query/key coordinates with RoPE; repeat the smaller KV head set for GQA; apply causal scaled dot-product attention. Each position sees only itself and preceding positions. PyTorch SDPA supplies the kernel.
4. `model/moe.py`: each token receives E router scores. Choose k experts, send only assigned tokens to each expert, weight their outputs, and sum them back into original token order. Every expert is a SwiGLU FFN: down(silu(gate(x)) * up(x)). Experts are learned functions, not manually assigned topics.
5. `model/transformer.py`: each block adds attention and FFN outputs through residual connections, with RMS normalization before each. Output logits reuse the input embedding matrix. Cross entropy trains all selected paths.
6. `training/loop.py`: average microbatch gradients, clip, update AdamW, evaluate held-out CE, and persist state. `expert_counts` in JSONL reveals assignment imbalance by layer; counts sum to batch * context * accumulation * k for each optimizer step.
7. `inference/generate.py`: choose one next token from final-position logits and append it. Temperature zero is greedy; top-k/top-p restrict sampling. This reference implementation recomputes the context each time.

## Router loss

For N tokens and E experts, let p[t,e] be full-softmax router probability. Let f[e] be assignments to expert e divided by N*k. Then auxiliary loss is E * sum_e(f[e] * mean_t(p[t,e])). The hard assignment fractions have no gradient; mean probabilities do. At uniform assignments/probabilities the value is 1. This encourages utilization balance, but does not guarantee specialization or perfectly equal counts. Average this scalar across layers and multiply by `load_balance_coef` before adding to training cross entropy.

For k>1 selected gate probabilities are renormalized to sum to one. For k=1 the original probability is retained so the task objective can update the router. No capacity factor or dropped tokens are used.

## Useful exercises

- Set `moe: null`, repeat an identical run, and compare parameters and validation curves.
- Set balancing coefficient to zero and inspect expert counts over steps.
- Change k from 2 to 1 and compare routing, active parameters, speed, and loss.
- Change KV heads while keeping query heads fixed; calculate cache storage for the future cache implementation.
- Inspect why held-out perplexity excludes router loss.

The next substantial work is a real dataset and controlled experiment, followed by KV-cache implementation. See AUDIT_AND_PLAN.md for acceptance criteria and primary references.
