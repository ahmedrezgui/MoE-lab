# Balancing-loss coefficient ablation

Final checkpoints use the same data, MoE architecture, seeds and training budget. Only the load-balancing coefficient changes within a seed. Full validation uses consecutive context-reset blocks, including the short tail.

| Coefficient | Mean full CE | Mean paired CE difference vs 0.01 | Sample SD of difference |
| ---: | ---: | ---: | ---: |
| 0 | 3.9617 | +0.0201 | 0.0108 |
| 0.01 | 3.9415 | +0.0000 | 0.0000 |
| 0.1 | 3.9278 | -0.0138 | 0.0131 |

Negative paired CE differences favor the displayed coefficient over 0.01. Seed spread is descriptive, not a significance test. Raw training objectives are not comparable because their auxiliary terms have different weights.

| Coefficient | Seed | Full CE | Sampled final CE | Early zero events | Late zero events | Loop seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 42 | 3.9634 | 4.0374 | 4 | 0 | 23.33 |
| 0 | 43 | 3.9897 | 4.0424 | 3 | 0 | 23.44 |
| 0 | 44 | 3.9319 | 3.9959 | 5 | 0 | 24.49 |
| 0.01 | 42 | 3.9365 | 4.0189 | 0 | 0 | 24.01 |
| 0.01 | 43 | 3.9820 | 4.0330 | 0 | 0 | 25.71 |
| 0.01 | 44 | 3.9061 | 3.9665 | 3 | 0 | 23.98 |
| 0.1 | 42 | 3.9377 | 4.0183 | 0 | 0 | 26.22 |
| 0.1 | 43 | 3.9587 | 4.0078 | 0 | 0 | 24.71 |
| 0.1 | 44 | 3.8869 | 3.9464 | 0 | 0 | 24.12 |

A zero event is one expert receiving no assignments in one optimizer step, summed across layers and steps in the window. It is not a count of permanently dead experts. Early/late windows are steps 1 through window_steps and the final window_steps; here window_steps=20 and final step=200.

## Per-layer routing shares

Shares count assignments, not unique tokens or gate weights. Layer IDs are logged module IDs, not block indices. Vectors follow zero-based expert order. Max/uniform is 1 for equal shares; it can reach 2 for four experts with top-2.

| Coefficient | Seed | Layer | Window | Assignment shares (%) | Max/uniform | Zero-step counts |
| ---: | ---: | --- | --- | --- | ---: | --- |
| 0 | 42 | 11 | early | 27.48, 27.07, 31.96, 13.50 | 1.278 | [0, 0, 0, 0] |
| 0 | 42 | 38 | early | 31.29, 38.72, 19.60, 10.38 | 1.549 | [2, 0, 1, 1] |
| 0 | 42 | 11 | late | 30.01, 26.49, 28.40, 15.10 | 1.200 | [0, 0, 0, 0] |
| 0 | 42 | 38 | late | 0.64, 37.45, 49.86, 12.05 | 1.994 | [0, 0, 0, 0] |
| 0 | 43 | 11 | early | 20.10, 20.58, 32.78, 26.54 | 1.311 | [0, 0, 0, 0] |
| 0 | 43 | 38 | early | 23.90, 23.65, 29.31, 23.14 | 1.173 | [0, 2, 0, 1] |
| 0 | 43 | 11 | late | 23.96, 22.15, 25.29, 28.61 | 1.144 | [0, 0, 0, 0] |
| 0 | 43 | 38 | late | 49.73, 16.91, 32.53, 0.83 | 1.989 | [0, 0, 0, 0] |
| 0 | 44 | 11 | early | 33.44, 18.38, 13.50, 34.68 | 1.387 | [0, 0, 0, 0] |
| 0 | 44 | 38 | early | 4.80, 21.38, 46.00, 27.82 | 1.840 | [3, 2, 0, 0] |
| 0 | 44 | 11 | late | 30.94, 27.90, 18.79, 22.37 | 1.238 | [0, 0, 0, 0] |
| 0 | 44 | 38 | late | 47.26, 0.29, 4.44, 48.01 | 1.920 | [0, 0, 0, 0] |
| 0.01 | 42 | 11 | early | 27.21, 22.98, 26.16, 23.66 | 1.088 | [0, 0, 0, 0] |
| 0.01 | 42 | 38 | early | 25.96, 23.32, 25.67, 25.05 | 1.039 | [0, 0, 0, 0] |
| 0.01 | 42 | 11 | late | 25.06, 22.65, 23.96, 28.33 | 1.133 | [0, 0, 0, 0] |
| 0.01 | 42 | 38 | late | 18.23, 24.14, 27.48, 30.15 | 1.206 | [0, 0, 0, 0] |
| 0.01 | 43 | 11 | early | 22.60, 23.31, 28.08, 26.01 | 1.123 | [0, 0, 0, 0] |
| 0.01 | 43 | 38 | early | 28.00, 25.01, 22.24, 24.76 | 1.120 | [0, 0, 0, 0] |
| 0.01 | 43 | 11 | late | 25.67, 24.21, 23.71, 26.41 | 1.056 | [0, 0, 0, 0] |
| 0.01 | 43 | 38 | late | 26.94, 24.57, 24.63, 23.86 | 1.078 | [0, 0, 0, 0] |
| 0.01 | 44 | 11 | early | 25.75, 22.00, 24.97, 27.28 | 1.091 | [0, 0, 0, 0] |
| 0.01 | 44 | 38 | early | 22.46, 22.65, 30.65, 24.25 | 1.226 | [1, 2, 0, 0] |
| 0.01 | 44 | 11 | late | 26.35, 27.71, 26.91, 19.02 | 1.108 | [0, 0, 0, 0] |
| 0.01 | 44 | 38 | late | 28.52, 23.45, 23.28, 24.74 | 1.141 | [0, 0, 0, 0] |
| 0.1 | 42 | 11 | early | 25.59, 24.74, 24.76, 24.91 | 1.024 | [0, 0, 0, 0] |
| 0.1 | 42 | 38 | early | 24.50, 22.71, 26.66, 26.13 | 1.066 | [0, 0, 0, 0] |
| 0.1 | 42 | 11 | late | 26.25, 23.36, 24.95, 25.44 | 1.050 | [0, 0, 0, 0] |
| 0.1 | 42 | 38 | late | 25.19, 24.97, 24.71, 25.13 | 1.007 | [0, 0, 0, 0] |
| 0.1 | 43 | 11 | early | 24.57, 23.59, 25.54, 26.30 | 1.052 | [0, 0, 0, 0] |
| 0.1 | 43 | 38 | early | 25.95, 23.97, 24.39, 25.69 | 1.038 | [0, 0, 0, 0] |
| 0.1 | 43 | 11 | late | 25.62, 24.43, 24.56, 25.39 | 1.025 | [0, 0, 0, 0] |
| 0.1 | 43 | 38 | late | 26.24, 24.03, 24.94, 24.79 | 1.050 | [0, 0, 0, 0] |
| 0.1 | 44 | 11 | early | 25.59, 23.78, 25.08, 25.55 | 1.024 | [0, 0, 0, 0] |
| 0.1 | 44 | 38 | early | 24.04, 24.22, 27.85, 23.89 | 1.114 | [0, 0, 0, 0] |
| 0.1 | 44 | 11 | late | 24.15, 24.66, 26.47, 24.71 | 1.059 | [0, 0, 0, 0] |
| 0.1 | 44 | 38 | late | 25.16, 24.79, 25.26, 24.79 | 1.010 | [0, 0, 0, 0] |

The accompanying ablation.json preserves per-step window counts, validation curves, full validation target counts, checkpoint/source/data hashes and resolved configurations. Raw metric logs are archived by run name. Balanced routing does not imply semantic specialization or better prediction quality. These validation data have been used during development; choosing a coefficient here needs confirmation on new held-out data.
