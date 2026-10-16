# Load-balancing coefficient ablation on TinyStories 1k

Nine fresh runs compared MoE load-balancing coefficients **0, 0.01 and 0.1** over seeds 42, 43 and 44. Within each seed, only the coefficient changed. The stronger coefficient produced more even routing and the lowest mean full-validation loss, but did not improve every seed.

| Coefficient | Mean full-validation CE | Mean paired CE difference vs 0.01 |
| --- | ---: | ---: |
| 0 | 3.9617 | +0.0201 |
| 0.01 | 3.9415 | 0 |
| 0.1 | 3.9278 | -0.0138 |

Lower CE is better. Coefficient zero was worse than 0.01 in all three pairs. Coefficient 0.1 improved over 0.01 for seeds 43 and 44 but was slightly worse for seed 42. Sample standard deviations of paired differences were 0.0108 for zero and 0.0131 for 0.1; these three-seed spreads are descriptive, not significance tests.

See [per-run quality and routing tables](results.md) and the [complete manifest](ablation.json). Raw metrics are archived in one subdirectory per run. The report can be regenerated from these archived records without checkpoints or GPU access.

## Fixed protocol

- Existing 1,000-training/100-validation-story TinyStories corpus; identical token stream/tokenizer fingerprints to the [previous experiment](../tinystories-1k-200/README.md). The shared BPE vocabulary contains 512 IDs including EOS 511.
- Same two-layer width-128 MoE, four experts of hidden width 341, top-2, context 128. Total parameters 1,213,056 and active estimate 689,280 do not vary with coefficient.
- Same AdamW settings, bf16 training, 200-step schedule with 10-step warmup, batch 2, accumulation 4. Each run processes 204,800 training targets. Initial weights and training samples are fixed within a seed; seeds differ across repetitions.
- Coefficients selected before results: 0 disables the auxiliary balancing objective, 0.01 is the current baseline, 0.1 weights that term ten times more strongly. Zero does not disable routing or task-loss gradients to the router.
- Fresh runs for all nine settings, including baseline. Order: [0, 0.01, 0.1] for seed 42; [0.01, 0.1, 0] for 43; [0.1, 0, 0.01] for 44. Each coefficient occupies each order position once. This does not control every laptop thermal/background effect.
- Separate sequential worker processes, each with a 180-second deadline for training plus evaluation. Validation input guard 100,000 tokens; no truncation. All workers completed on the RTX 3050 Ti Laptop GPU with no cloud use.
- Final step-200 weights scored in fp32 on all 28,667 next-token targets, in consecutive context-reset blocks with tail included, as in the [full-stream evaluation policy](../tinystories-1k-full-validation/README.md). The worker evaluates the returned final model immediately after saving last.pt; it does not evaluate selected best weights.
- Routing windows fixed at steps 1-20 and 181-200. Each window contains 40,960 assignments per layer. Raw training objectives differ with coefficient and are not the quality comparison; full CE excludes the auxiliary term.

## Routing findings

With no auxiliary balancing term, logged layer 38's busiest expert receives approximately 48.0-49.9% of late-window assignments in each seed, close to the 50% maximum for top-2 among four experts. That is concentrated traffic, even though all experts still receive some assignments. The late max/uniform ratio in that layer ranges from approximately 1.92 to 1.99.

At 0.01, late layer ratios range from about 1.06 to 1.21 across layers/seeds. At 0.1 they range from about 1.01 to 1.06, close to equal traffic. Stronger balancing therefore changed assignment distributions substantially under these conditions.

Early zero-assignment expert-step events totaled 12 across the zero-coefficient runs, 3 across baseline runs and 0 with 0.1. All settings had zero such events in the late windows. An event is one expert receiving no assignments in one step, not a unique permanently unused expert. This shows why both traffic shares and zero counts matter: zero events can disappear while concentration remains high.

More even traffic was associated with a slightly better mean validation score here, but the strongest coefficient was not best for every seed. The ablation controls the objective weight, not the semantic role of each expert; neither balanced counts nor a small CE difference proves expert specialization. Expert identities are not aligned across seeds.

## Interpretation and limits

The evidence supports retaining an auxiliary balancing objective for this tiny setup and treating 0.1 as a candidate for confirmation. It does not establish a universal optimal coefficient or justify changing defaults based only on this validation sweep. These validation stories have already been used during development; trying values and choosing the best introduces selection bias. Confirm a candidate on untouched data and, if useful, additional paired seeds.

Only three coefficient values and three seeds were evaluated over 200 updates on a selected corpus prefix. There is no significance test, semantic specialization evaluation or deployment-quality claim. Training-loop durations include evaluation/logging/checkpoint I/O, and zero coefficient still computes routing and auxiliary statistics, so this is not an overhead-removal benchmark. Fresh baseline results can differ slightly from older runs because deterministic CUDA execution is not enforced.

## Reproduce

With the existing prepared corpus, choose fresh output directories:

```powershell
.venv/Scripts/python scripts/run_balance_ablation.py --data-dir data/tinystories-1k-eos --output-dir checkpoints/balance-ablation-1k --coefficients 0 0.01 0.1 --seeds 42 43 44 --steps 200 --window-steps 20 --device cuda --timeout-seconds 180 --max-val-tokens 100000
.venv/Scripts/python scripts/report_balance_ablation.py --ablation-dir checkpoints/balance-ablation-1k --output-dir reports/reproduced-balance
```

For report-only regeneration from tracked evidence:

```powershell
.venv/Scripts/python scripts/report_balance_ablation.py --ablation-dir reports/balance-ablation-1k --output-dir reports/recomputed-balance
```

The manifest records source hashes at experiment launch, config snapshots, data hashes, environment, full target coverage, checkpoint hashes, paired differences and per-step routing in both windows. Git HEAD is the preceding commit with dirty state true. The report-rendering script was added after workers launched; it does not affect training. Local job files, full evaluation files, logs and checkpoints remain under checkpoints/balance-ablation-1k. Raw corpus and model binaries remain ignored.

## Verification

118 tests passed, including coefficient-independent initial weights, a real isolated CPU sweep, equal token budgets, full target coverage, paired summary arithmetic, invalid argument rejection, output protection and worker timeout handling. All nine actual runs had identical initial sampled CE within each seed. Configs were checked to differ only in coefficient within seed; all checkpoint/log hashes, 204,800 training targets, 28,667 validation targets and 40,960 assignments per layer/window were verified. Lint/format passed. The published Markdown was regenerated from the stored manifest and matched exactly.
