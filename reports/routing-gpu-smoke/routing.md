# Expert-utilization report

Percentages count hard expert assignments during training, summed over the selected steps and microbatches. Each layer sums to 100%. With top-2, each token contributes two assignments; these are not percentages of unique tokens or weighted router outputs.

Layer IDs are the recorded module IDs, not decoder block numbers. A max/uniform ratio of 1 means equal assignment shares. With four experts, 1.6 means the busiest expert receives 40% of assignments versus the uniform 25%. No automatic collapse threshold is imposed.

## checkpoints/comparison-gpu-smoke/moe-seed-42/metrics.jsonl

Observed steps 1–20 (20 steps). Source SHA-256: `a062bb29c238a28e3b636542a77dac67ddfd37a1eec751d98d82d569a691d376`.

| Layer ID | Expert | Assignments | Share (%) | Steps with zero assignments |
| --- | ---: | ---: | ---: | ---: |
| 11 | 0 | 7,958 | 19.43 | 0 |
| 11 | 1 | 5,942 | 14.51 | 0 |
| 11 | 2 | 11,716 | 28.60 | 0 |
| 11 | 3 | 15,344 | 37.46 | 0 |
| 38 | 0 | 10,583 | 25.84 | 0 |
| 38 | 1 | 6,609 | 16.14 | 1 |
| 38 | 2 | 12,034 | 29.38 | 3 |
| 38 | 3 | 11,734 | 28.65 | 0 |

| Layer ID | Aggregate max/uniform | Highest step max/uniform |
| --- | ---: | ---: |
| 11 | 1.498 | 1.998 |
| 38 | 1.175 | 2.000 |

## checkpoints/comparison-gpu-smoke/moe-seed-43/metrics.jsonl

Observed steps 1–20 (20 steps). Source SHA-256: `4bd31d00f21a3748cd15d76bc24e5798d26a2a8eff46897594db9d3f6fdcdd1a`.

| Layer ID | Expert | Assignments | Share (%) | Steps with zero assignments |
| --- | ---: | ---: | ---: | ---: |
| 11 | 0 | 9,148 | 22.33 | 0 |
| 11 | 1 | 13,869 | 33.86 | 0 |
| 11 | 2 | 9,750 | 23.80 | 0 |
| 11 | 3 | 8,193 | 20.00 | 0 |
| 38 | 0 | 10,112 | 24.69 | 1 |
| 38 | 1 | 13,600 | 33.20 | 0 |
| 38 | 2 | 8,810 | 21.51 | 1 |
| 38 | 3 | 8,438 | 20.60 | 1 |

| Layer ID | Aggregate max/uniform | Highest step max/uniform |
| --- | ---: | ---: |
| 11 | 1.354 | 1.945 |
| 38 | 1.328 | 2.000 |

Aggregate shares can hide changes over time. The accompanying JSON keeps every step's counts and percentages; use step windows to compare early and late routing. Zero assignments indicate no dispatched work in an observed step, not necessarily a dead expert outside this window. Balanced traffic does not prove useful specialization or better language quality. These counts do not measure validation routing or time spent by each expert.
