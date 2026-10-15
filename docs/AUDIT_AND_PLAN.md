# Audit and implementation plan

Audit date: 2026-10-15. Target: an understandable, locally runnable MoE language model and an evidence-based portfolio project.

## Initial findings

| Priority | Finding | Consequence |
|---|---|---|
| Blocker | Attention, MoE, Transformer, BPE, training, generation, and perplexity were stubs | No trainable or usable model existed |
| High | All model tests were skipped; two contained only ellipses | Green config tests could not establish model correctness |
| High | Dataset length excluded a valid start and did not reject short input | Boundary bugs in the data path |
| High | A test used model_copy with a raw MoE dictionary | Pydantic does not validate model_copy updates; unskipping would fail |
| High | tiny-moe claimed matched active parameters, but uses two full-size experts versus one dense FFN | Invalid dense/MoE comparison |
| Medium | Config accepted unknown keys and odd rotary head dimensions | Typos or invalid shapes could survive until training |
| Medium | README advertised quantization, platform integrations, and CI as though operational | Portfolio claims exceeded available evidence |
| Medium | Setup omitted torch; smoke used a large bf16 configuration and unimplemented data preparation | No beginner-friendly local quickstart |
| Medium | No Git metadata in this workspace | Cannot verify repository history, remote CI runs, or create reviewable commits yet |

Existing useful pieces: typed YAML configs, package structure, MLflow wrapper, FastAPI health/metrics, Docker skeleton, and a lint/config-test workflow file. No AGENTS.md instructions were found in the project or checked parent locations.

## Milestones and acceptance criteria

1. **Runnable educational baseline (implemented in this iteration).** Byte-level BPE, deterministic text preparation, correct memmap blocks, causal RoPE/GQA, RMSNorm, SwiGLU, dropless top-k MoE, balancing loss, tied output embeddings, dense baseline. Test causal independence, sparse/dense dispatch equivalence, gradients, and one-batch overfitting.
2. **Training and local use (implemented in this iteration).** AdamW, cosine decay/warmup, gradient accumulation/clipping, CUDA mixed precision, CPU fp32, CE-only validation perplexity, JSONL routing counts, checkpoint/RNG resume, CLI sampling, model-backed API. Verify CPU resume equivalence and a local NVIDIA smoke run. The toy corpus establishes plumbing, not general language ability.
3. **Real corpus and controlled experiment (next).** Obtain a licensed corpus with documented provenance (candidate: TinyStories); keep document-level train/validation separation and deduplicate. Train BPE on training documents only. The simple reference BPE needs a bounded training sample or a faster backend before scaling. Add explicit EOS/document boundary policy. Start with a small subset and the 4 GB GPU preset; record tokens seen, memory, throughput, validation CE, and sample generations. Compare dense/MoE at a declared active-parameter or compute budget, adjusting FFN widths as needed. Repeat seeds and balance-loss coefficients; plot per-layer expert assignment fractions. Acceptance: reproducible configs and evidence, no claim that MoE must outperform dense.
4. **Inference experiment.** Implement per-layer KV cache with RoPE offsets and correct causal masks. Compare logits against uncached inference including chunked prefill; benchmark prefill/decode separately. Add streaming and cancellation/backpressure. Only investigate quantization after a quality baseline; custom weights are not automatically compatible with GGUF, llama.cpp, or vLLM.
5. **Deployment.** First test Docker locally with a mounted checkpoint/tokenizer. Select Azure hosting based on measured CPU/GPU inference requirements, available quota, and current cost. Add infrastructure configuration, artifact versioning, readiness, request limits, authentication, and deployment smoke tests before exposing a public demo. No Azure resources or paid training have been created.
6. **Portfolio release.** Initialize/link the intended Git repository, run CI remotely, document architecture and routing math, publish a reproducible experiment report with plots and limitations, then a short demo. Add MLflow/DVC only when experiment/artifact volume justifies them. A resume bullet should contain measured model size, corpus size, perplexity, and latency; leave results blank until measured.

## Local hardware

Detected NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB VRAM. Begin with batch size 2, context 128, d_model 128, two layers and four top-2 experts. This is a starting point, not a guaranteed maximum. Increase only after profiling. Use gradient accumulation to increase effective batch without growing the microbatch. CPU tests use the smaller byte-vocabulary preset.

## Design choices and limitations

- All model weights start random; no pretrained model or LLM API. PyTorch supplies autograd, linear algebra, and SDPA kernels.
- Routing evaluates only selected token/expert pairs; Python loops favor readability over production throughput. No token dropping, capacity limit, distributed experts, or fused dispatch.
- For top-k > 1, selected weights are normalized. For top-1, retain gate probability to preserve task-loss router gradients.
- Auxiliary loss is E * sum(assignment_fraction * mean_router_probability), where fractions count all k assignments and sum to one. Average across layers, then multiply by the configured coefficient. Validation perplexity excludes this term.
- Active-parameter counts include shared weights and k expert FFNs per layer. They are an architectural estimate, not measured FLOPs or bytes accessed.
- The reference sampler recomputes a sliding window and resets its positions. KV cache remains a future milestone.
- Exact resume equivalence is tested on CPU within the same environment. CUDA kernels, software changes, or different hardware may introduce numerical differences.
- Local API is synchronous and serializes model access. Public-hosting hardening and SSE remain future work.

## Primary references

- [Switch Transformers paper](https://www.jmlr.org/papers/v23/21-0998.html): balancing objective; this implementation explicitly adapts assignment fractions to top-k.
- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention): causal masking and disabling attention dropout in evaluation.
- [PyTorch installation guide](https://pytorch.org/get-started/locally/): choose and verify a CUDA-enabled build for local GPU use.

## Recorded evidence

See [local verification results](LOCAL_RESULTS.md) for the completed GPU/CPU runs and [Azure student feasibility](AZURE_FEASIBILITY.md) for the account/price assessment. The user prefers Azure only if their student plan supports affordable GPU compute and it is worthwhile compared with local training.
