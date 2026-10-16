# Longer bounded comparison: validation and routing

Early routing: steps [1, 20]. Late routing: steps [181, 200]. CE is measured at each window's final step, not averaged over its training losses.

| Model | Seed | Early CE | Final CE | Change | Best CE (step) | Loop seconds | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| dense | 42 | 5.4101 | 3.9018 | -1.5082 | 3.9018 (200) | 15.08 | 34.98 |
| moe | 42 | 5.4409 | 4.0189 | -1.4220 | 4.0189 (200) | 25.76 | 45.17 |
| moe | 43 | 5.4302 | 4.0330 | -1.3972 | 4.0330 (200) | 25.04 | 45.17 |
| dense | 43 | 5.4144 | 3.9095 | -1.5049 | 3.9095 (200) | 11.28 | 34.98 |
| dense | 44 | 5.4172 | 3.8705 | -1.5467 | 3.8705 (200) | 10.97 | 34.98 |
| moe | 44 | 5.4358 | 3.9667 | -1.4691 | 3.9667 (200) | 23.96 | 45.17 |

## Paired final validation differences

Positive MoE-minus-dense CE favors dense; negative favors MoE.

| Seed | MoE minus dense CE |
| --- | ---: |
| 42 | +0.1170 |
| 43 | +0.1235 |
| 44 | +0.0962 |

Mean paired difference: +0.1122. Sample standard deviation: 0.014287146246353391. This spread is descriptive, not a significance test.

## Early and late expert assignments

Shares are percentages of hard assignments (top-2 gives two per token). Layer IDs are logged module IDs. Expert order in each vector is 0, 1, 2, 3. Max/uniform = 1 means equal traffic. Zero-step counts are per expert within the window, not the number of experts unused for the entire run.

| Seed | Layer | Window | Expert shares (%) | Max/uniform | Zero-step counts |
| ---: | --- | --- | --- | ---: | --- |
| 42 | 11 | early | 27.21, 22.98, 26.16, 23.66 | 1.088 | [0, 0, 0, 0] |
| 42 | 38 | early | 25.96, 23.32, 25.67, 25.05 | 1.039 | [0, 0, 0, 0] |
| 42 | 11 | late | 25.06, 22.65, 23.96, 28.33 | 1.133 | [0, 0, 0, 0] |
| 42 | 38 | late | 18.23, 24.14, 27.48, 30.15 | 1.206 | [0, 0, 0, 0] |
| 43 | 11 | early | 22.60, 23.31, 28.08, 26.01 | 1.123 | [0, 0, 0, 0] |
| 43 | 38 | early | 28.00, 25.01, 22.24, 24.76 | 1.120 | [0, 0, 0, 0] |
| 43 | 11 | late | 25.67, 24.21, 23.71, 26.41 | 1.056 | [0, 0, 0, 0] |
| 43 | 38 | late | 26.94, 24.57, 24.63, 23.86 | 1.078 | [0, 0, 0, 0] |
| 44 | 11 | early | 25.75, 22.00, 24.97, 27.28 | 1.091 | [0, 0, 0, 0] |
| 44 | 38 | early | 22.46, 22.65, 30.65, 24.25 | 1.226 | [1, 2, 0, 0] |
| 44 | 11 | late | 26.40, 27.72, 26.91, 18.96 | 1.109 | [0, 0, 0, 0] |
| 44 | 38 | late | 28.48, 23.59, 23.34, 24.59 | 1.139 | [0, 0, 0, 0] |

The accompanying analysis JSON includes all recorded validation evaluations and per-step routing data for both windows. Aggregate traffic can hide transient concentration. Neither balanced traffic nor a CE change demonstrates semantic specialization or a causal benefit from balancing loss. Timing includes evaluation, logging and checkpoint I/O; memory covers the PyTorch allocator, not total GPU usage.
