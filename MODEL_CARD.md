# MoE Lab: portfolio model card

## Purpose and identity

This is a small decoder-only language model trained from random initialization to explore mixture-of-experts routing. Its intended use is local architecture exploration, reproducible experiments, and a portfolio demonstration. It continues text; it is not a chat assistant or instruction-tuned model.

The demo selects **coefficient 0.1, training seed 42, final step 200** from the recorded ablation. Seed 42 is the first declared seed, not the seed with the lowest confirmation loss. No new training or checkpoint selection was performed for the UI.

- Local checkpoint: `checkpoints/balance-ablation-1k/coef-0.1-seed-42/last.pt`
- SHA-256: `39b93d9e7b23bf2cf8eef8fa7e6d85fa02d6ab18e67494c9bb097e521916f103`
- Weights are not bundled in Git. The local checkpoint and its adjacent `tokenizer.json` are required, or reproduce training below.
- Implementation: from-scratch attention, tokenizer, routing, model and training code using PyTorch primitives. Gradio is the presentation layer; no hosted LLM API or downloaded pretrained model is used.

## Architecture and training

| Property | Value |
| --- | --- |
| Layers / residual width | 2 / 128 |
| Query heads / key-value heads | 4 / 2 (GQA) |
| Position encoding / normalization | RoPE / RMSNorm |
| Feed-forward block | SwiGLU, four experts per layer, top-2 routing |
| Hidden width per expert | 341 |
| Total / estimated active parameters | 1,213,056 / 689,280 |
| Vocabulary / context | 512 byte-BPE entries including EOS 511 / 128 tokens |
| Input/output embeddings | Tied |
| Balancing coefficient | 0.1 |
| Optimizer / training precision | AdamW / bf16 autocast, fp32 weights |
| Steps / processed targets | 200 / 204,800 |
| Microbatch / accumulation | 2 / 4 |
| Training seed | 42 |

Training data are the first 1,000 training stories of pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. BPE fitting uses only a bounded 65,536-byte training prefix. Documents undergo declared normalization/exact deduplication and receive EOS boundaries without an attention barrier. The prepared training stream has 412,068 tokens; random block sampling with replacement means 204,800 processed targets are not a complete pass through all training data.

The source card declares CDLA-Sharing-1.0. Keep the original card/license notice with downloaded artifacts. See [dataset provenance and preparation](reports/tinystories-1k-200/README.md) for the source and transformations. This card does not assign a new license to the dataset or model weights.

## Measured evidence

The selected seed-42 checkpoint has full development CE **3.9377** and confirmation CE **3.8954**, under the same segmented full-stream policy but different stories. Development uses the first 100 validation stories; confirmation uses the next 200, checked for normalized exact overlaps against training/development and against each other. Near-duplicates are not excluded.

Across three paired seeds, coefficient 0.1 averaged confirmation CE 3.8849 versus 3.8986 at 0.01. Two seeds favored the candidate, while seed 42 was effectively tied. The candidate was selected on development evidence before confirmation scoring. These small experiments do not establish statistical significance, universal coefficient superiority, or semantic expert specialization. The confirmation stories have now been observed and should not be reused for further tuning while described as untouched.

- [Dense/MoE experiments](reports/tinystories-1k-200/README.md)
- [Full-stream evaluation policy](reports/tinystories-1k-full-validation/README.md)
- [Balancing-coefficient ablation](reports/balance-ablation-1k/README.md)
- [Reserved-story confirmation](reports/balance-confirmation/README.md)

## Actual generation quality

All six outputs from three fixed prompts, each decoded greedily and with temperature 0.8, are preserved in [demo-samples.json](reports/demo-samples.json). They use the selected checkpoint, CPU fp32, a 64-new-token cap, top-k 40, top-p 0.95 and sampling seed 42. No output was discarded or selected for fluency. Greedy decoding ignores top-k/top-p. GPU output can differ from CPU output; recorded hashes identify the exact weights.

Representative greedy output begins:

> The little fox car car car car car car car car car car car car car car car car car car car car car

This repetition is a real limitation of the trained model, not a polished story. Stochastic outputs also contain broken words and weak coherence. All six recorded samples reached the token cap rather than predicting EOS. Loss reduction and correct EOS support do not guarantee the model has learned useful stopping behavior or fluent generation.

## Local playground

Install the optional UI dependency in the existing environment (install the intended CUDA PyTorch build first, as in the README):

```powershell
.venv/Scripts/python -m pip install -e ".[ui]"
.venv/Scripts/python scripts/demo_ui.py
```

Open http://127.0.0.1:7860. The default device is `auto`: CUDA if available, otherwise CPU. Force CPU with `--device cpu`; choose another trained model with `--checkpoint PATH`; change the port with `--port 7861`. Keep the matching tokenizer beside its checkpoint. The UI validates vocabulary and, for current checkpoints, the tokenizer's training fingerprint. Gradio 6.30.0 is pinned for the optional UI extra.

The interface provides prompt examples, maximum new tokens, temperature, top-k, top-p, a sampling seed, full generated text, token counts and stopping reason. A dedicated random generator per request makes repeated settings reproducible on the same runtime without changing global RNG state. Requests are serialized through a queue and model lock. The checkpoint is loaded once at launch; this interface does not upload files or train models.

The app binds to loopback only and disables Gradio public sharing. It runs locally and does not require an Azure resource or a paid inference API. This local demo is not an authenticated public deployment. Stop the terminal process with Ctrl+C. There is no KV cache or streaming output; the model recomputes a sliding context during generation. Prompts longer than 128 tokens are rejected, and the output cap is 256 new tokens.

## Reproduce the checkpoint and samples

Prepare the pinned corpus using the commands in the recorded 1k-story experiment, then train a fresh run:

```powershell
.venv/Scripts/python scripts/train.py --config configs/training/laptop.yaml --model configs/model/portfolio-moe.yaml --data-dir data/tinystories-1k-eos --device cuda --output-dir checkpoints/portfolio --max-steps 200
.venv/Scripts/python scripts/demo_ui.py --checkpoint checkpoints/portfolio/last.pt --device cuda
.venv/Scripts/python scripts/demo_samples.py --checkpoint checkpoints/portfolio/last.pt --device cpu --output reports/my-demo-samples.json
```

Fresh GPU training is not guaranteed to recreate bit-identical weights or files. The sampling seed is separate from the training seed. Samples are repeatable for a fixed checkpoint/device/runtime; use the stored checkpoint hash to identify the exact artifact. Sample export refuses to overwrite an existing file.

## Limits and suitable portfolio claims

Suitable claim: implemented and tested an educational MoE language model, controlled parameter budgets and paired experiments, audited expert traffic, confirmed a candidate on reserved data, and built a local interactive inference demo with traceable weights and settings.

Unsupported claim: a fluent assistant, factual knowledge system, competitive pretrained LLM, proof that MoE is faster/better than dense, or a generally optimal balancing coefficient. Generated text can be incoherent, fabricated or undesirable; there has been no factuality, safety or instruction-following evaluation. The source is synthetic English stories, not representative multilingual or domain data. Outputs are for inspection, not decision-making.


## Larger training candidate (2026-10-16)

A separate candidate keeps this architecture, coefficient 0.1, seed 42 and byte-identical tokenizer/development stream, while increasing training to 5,000 stories and 2,000 fresh optimizer updates. Its SHA-256 is `a0aa2ab93c9628b75ddd81fc91dcba80b0cf8cb7f39686828628035c15b2b10b`; weights are local at `checkpoints/story-5k-2k/last.pt`. The original default checkpoint and original samples above remain unchanged.

Full CPU fp32 development CE improved from 3.937743 to 2.198544 (perplexity 51.3027 to 9.0119), scoring the same 28,667 targets. The local training loop took 237.11 seconds, with 45.17 MiB peak allocated and 50.00 MiB peak reserved PyTorch memory. These are allocator figures, not total GPU usage. The idle original UI and concurrent test execution limit runtime comparisons.

The [full experiment report](reports/story-5k-2k/README.md) contains commands, controls, fingerprints, early/late routing and every [before](reports/story-5k-2k/samples-before.json)/[after](reports/story-5k-2k/samples-after.json) output. Recognizable phrases improve, but repetitions, inconsistent pronouns and incoherence remain; all six outputs still reach the cap without EOS. Both corpus size and training horizon change, so their separate contributions are unknown. This is one seed on an inspected development set, with no new confirmation/test-set claim.

Launch with `.venv/Scripts/python scripts/demo_ui.py --checkpoint checkpoints/story-5k-2k/last.pt --port 7861`. The candidate is an educational improvement, not a fluent assistant.
