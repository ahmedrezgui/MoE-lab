# Full-stream validation of the 1k-story experiment

The dense model retained lower validation CE in all three paired seeds after scoring the entire prepared validation stream. Mean full-stream CE was **3.8301 dense versus 3.9415 MoE**. The mean MoE-minus-dense gap was +0.1114, with sample standard deviation 0.0207 across three pairs. These are descriptive observations for this corpus and context policy, not a significance test or universal model ranking.

The [machine-readable evidence](full-validation.json) records checkpoint/data/source fingerprints, scoring coverage, summed negative log likelihood, CE/perplexity, environment and evaluation duration. The [original experiment](../tinystories-1k-200/README.md) and [training manifest](../tinystories-1k-200/comparison.json) define the model configurations and training protocol. No model was trained further or selected using these new scores.

## Coverage and scoring policy

- Use `last.pt` at step 200 for all six runs. The equal training budget remains 204,800 targets per model/seed. Best-checkpoint selection is not repeated.
- The validation stream contains 28,668 token IDs. Every target at positions 1 through 28,667 is scored once, for 28,667 targets per checkpoint. Position zero is excluded because there is no preceding token or dedicated BOS token.
- Partition targets into 223 full blocks of 128 plus a final block of 123. A block starting at offset s receives inputs `ids[s:s+length]` and predicts `ids[s+1:s+length+1]`. The boundary token can be a target of one block and an input to the next; it is never scored twice.
- Context resets at every block. The first target of each block has one input token of context, later targets have progressively more. This is full target coverage under a segmented policy, **not** a stride-one/max-context sliding-window score.
- Blocks remain packed across document boundaries. EOS and the first token after EOS are scored normally; no attention reset or loss mask is added at document boundaries.
- fp32 evaluation, no autocast/dropout, no router auxiliary loss. Batch size is two for full blocks; the short tail is scored separately without padding. Sum all target negative log likelihoods and divide by the exact target count; do not average block means or batch perplexities.
- Original validation/tokenizer hashes and checkpoint config/step/data metadata must match the training manifest. A maximum-token guard rejects oversized files without truncation. Run outputs require a new directory and retain a failed status if a checkpoint fails.

## Results

| Model | Seed | Prior sampled CE | Full-stream CE | Full-stream perplexity | Evaluation seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 42 | 3.9018 | 3.8285 | 46.00 | 0.611 |
| moe | 42 | 4.0189 | 3.9365 | 51.24 | 1.121 |
| moe | 43 | 4.0330 | 3.9820 | 53.62 | 0.910 |
| dense | 43 | 3.9095 | 3.8484 | 46.92 | 0.497 |
| dense | 44 | 3.8705 | 3.8133 | 45.30 | 0.477 |
| moe | 44 | 3.9667 | 3.9059 | 49.70 | 0.960 |

Paired full-stream MoE-minus-dense differences are +0.1080 (seed 42), +0.1336 (43), and +0.0926 (44). Dense remains lower for every pair. This strengthens the observation beyond the original fixed 1,024-target sample, but the 100 validation stories are still a prefix from one source and only three initialization seeds were tested.

The same frozen checkpoints have different scores under the sampled and full-stream protocols. The token positions, weighting, and block-context alignments differ. Lower full-stream CE is **not additional learning**, and these scores should not be plotted as a continuation of the training loss curve. These are validation data already used during development, not a new untouched test set.

Evaluation duration includes forward scoring, token transfers and loss accumulation with CUDA synchronization at both boundaries. It excludes model/checkpoint loading, hashing and input validation. Runs occur sequentially in one process; early runs may incur kernel/cache startup costs. Subsecond timings are not a reliable inference-speed benchmark. This evaluation does not remeasure training memory or write checkpoints.

## Reproduce

With the prior local checkpoints and prepared data present, choose a fresh output directory:

```powershell
.venv/Scripts/python scripts/evaluate_full.py --comparison-dir checkpoints/comparison-1k-200 --data-dir data/tinystories-1k-eos --output-dir reports/reproduced-full-validation --device cuda --batch-size 2 --max-tokens 100000
```

For CPU scoring use `--device cpu`; weights and scores remain fp32. `--max-tokens` is an input-size guard, not a sample size: files above the bound fail. The default is 1,000,000 tokens. The new command produces `full-validation.json`; this explanatory Markdown records the completed run. Checkpoints and corpus files remain ignored by Git.

The full report can be audited from saved counts and summed losses, but rescoring requires the matching checkpoints and token files. The manifest records checkpoint SHA-256 hashes; numeric GPU reproducibility is not guaranteed across software/hardware changes. The original sampled metrics and routing reports were preserved unchanged.

## Verification and next step

108 tests passed. New tests explicitly cover first/last target coverage, nondivisible tails, token-weighted loss, batch-size invariance, model-mode restoration, invalid data/precision, provenance rejection, and real tiny checkpoint evaluation on CPU. All six GPU records scored exactly 28,667 targets and match the recorded checkpoint and data fingerprints. Lint and formatting passed.

A useful next experiment is a controlled balancing-loss coefficient ablation: keep data, model, seed and training budget fixed, change the coefficient, then compare routing and this full-stream validation policy. This would test the role of the balancing objective rather than assuming balanced traffic causes better predictions.
