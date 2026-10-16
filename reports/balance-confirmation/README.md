# Confirmation on previously unused validation stories

Coefficient **0.1** retained a modest mean advantage over **0.01** on a newly reserved block of stories. Mean confirmation CE was **3.8849 versus 3.8986**, a paired difference of **-0.01367**. Seeds 43 and 44 favor 0.1; seed 42 is effectively tied (difference about -0.000014). This supports the candidate in this small experiment, not a universal coefficient choice or statistical significance claim.

The [complete scoring record](confirmation.json), [preparation/overlap record](preparation-metadata.json), and [download provenance](download-manifest.json) preserve coverage, source/checkpoint fingerprints and selection policy. The [original ablation](../balance-ablation-1k/README.md) selected these two candidates before this confirmation block was scored. No additional training or best-checkpoint selection took place.

## Selection fixed before scoring

- Same pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- Existing development set: first 100 validation stories. Confirmation candidates: stories 101-300 (zero-based start 100, count 200), downloaded as part of a bounded 300-story prefix.
- The first 100 normalized source documents must exactly match the old development split. Prior train/development file hashes must match the original preparation metadata, and all original token/tokenizer artifacts must retain their original hashes.
- Normalize CRLF/CR to LF and strip outer whitespace, then exclude exact matches against all 1,000 original training documents, all 100 development documents, and duplicates within the fixed candidate slice. Drop excluded records without replacing them from later stories.
- All 200 candidates were retained: zero normalized exact overlaps with training/development and zero internal exact duplicates. This does not detect semantic or near-duplicates, and the source prefix is not a random sample.
- Reuse the exact training tokenizer bytes (vocabulary 512, EOS 511). Never fit merges on the confirmation text. Append one EOS after each retained document, preserving the training boundary policy.

The resulting stream contains 80,139 tokens. Every checkpoint scores all 80,138 available next-token targets: 626 full blocks of 128 and a 10-target tail. Position zero has no preceding context and is excluded. Context resets at block boundaries; EOS and cross-document transitions remain included. This is the same full-stream scoring policy used in development, not a maximum-context sliding-window score.

## Paired confirmation results

| Seed | Coefficient 0.01 CE | Coefficient 0.1 CE | Candidate minus baseline |
| ---: | ---: | ---: | ---: |
| 42 | 3.895440 | 3.895426 | -0.000014 |
| 43 | 3.933527 | 3.911754 | -0.021773 |
| 44 | 3.866771 | 3.847549 | -0.019222 |

Mean paired difference: -0.013670; sample standard deviation: 0.011895 across three seeds. Negative favors 0.1. The tiny seed-42 difference should be treated as a practical tie, not emphasized as another clear win. No confidence interval, significance test, or language-quality threshold is claimed.

These are the same final step-200 checkpoints from the ablation, with checkpoint hashes verified before evaluation. All six models were scored in fp32 without auxiliary loss. New confirmation CE should not be interpreted as a continuation of training or compared with development CE as if the difference represented learning: the stories differ. The candidate-versus-baseline comparison is valid within the common confirmation set and policy.

## What the result establishes

The improvement observed during coefficient selection persisted on this separate exact-disjoint document block, with a similar mean paired gap. This reduces concern that the original observation depended only on the first 100 development stories. It does not eliminate uncertainty from only three seeds, a single synthetic-story source, possible near-duplicates, fixed context segmentation, or a short 200-step training budget.

The block was unused by this project's training, tokenizer fitting and coefficient selection before this evaluation. Now that the result has been inspected, it is no longer untouched. Do not repeatedly tune coefficients against this block and continue describing it as fresh confirmation. Candidate 0.1 is supported for the current small setup; the configured default is not silently changed by an evaluation command.

## Reproduce

Use fresh artifact directories, or adjust paths when these already exist:

```powershell
.venv/Scripts/python scripts/download_data.py --output data/raw/tinystories-confirmation-prefix --train-documents 1000 --val-documents 300
.venv/Scripts/python scripts/prepare_confirmation.py --source data/raw/tinystories-confirmation-prefix/val.jsonl --prior-train data/raw/tinystories-1k/train.jsonl --prior-val data/raw/tinystories-1k/val.jsonl --training-data-dir data/tinystories-1k-eos --output data/tinystories-confirmation --start-document 100 --document-count 200
.venv/Scripts/python scripts/evaluate_confirmation.py --ablation-dir checkpoints/balance-ablation-1k --heldout-dir data/tinystories-confirmation --output-dir reports/reproduced-confirmation --device cuda --batch-size 2 --max-tokens 200000
```

The downloader's original source card/license notice remains with local raw data. `prepare_confirmation.py` writes a frozen tokenizer, `ids_confirmation.bin`, and metadata; the evaluator accepts only the predeclared 0.01/0.1 pair from a complete ablation. It checks checkpoint hashes and saved config/step/data metadata, preserves failed status on scoring errors, and rejects oversized inputs rather than truncating them. Its output is `confirmation.json`; this Markdown documents the completed interpretation. CPU evaluation is available with `--device cpu`. The original training, sampled/full development reports, and model files remain unchanged.

Runtime measurements include scoring/transfers but exclude loading/hashing; they are short sequential evaluations and not a production-speed benchmark. No cloud services or paid compute were used. Local corpus and checkpoints remain ignored by Git; public evidence contains metadata and scores, not story text or model weights.

## Verification and next milestone

**125 tests passed**, with lint/format checks passing. New tests cover frozen tokenization, exact overlap removal, duplicate normalization, prefix/source provenance rejection, missing candidates, artifact/checkpoint changes, paired arithmetic, and end-to-end tiny-checkpoint CPU scoring. The actual six records were checked against checkpoint/source/data hashes and exact target coverage; normalized document sets were independently compared for overlap.

A useful next milestone is packaging a reproducible local generation demo and model card using the documented experiment, with fixed prompts and candid limitations. More tuning would require a new confirmation design rather than repeated selection on these now-observed stories.
