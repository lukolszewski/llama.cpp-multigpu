| Workload | Context | Upstream llama.cpp | llama.cpp-multigpu | Improvement |
| --- | --- | --- | --- | --- |
| 1 slot - prefill | 5k | 398 | 390 | 0.98x (-7 t/s) |
| 1 slot - generate | 5k | 34.2 | 37.6 | 1.10x (+3.4 t/s) |
| 1 slot - prefill | 50k | 316 | 348 | 1.10x (+32 t/s) |
| 1 slot - generate | 50k | 23.6 | 31.7 | 1.34x (+8.1 t/s) |
| 1 slot - prefill | 150k | 227 | 263 | 1.16x (+36 t/s) |
| 1 slot - generate | 150k | 14.6 | 27.2 | 1.86x (+12.6 t/s) |
| 1 slot - prefill | 200k | not run | 238 | n/a |
| 1 slot - generate | 200k | not run | 26.0 | n/a |
| 1 slot - prefill | 250k | not run | 217 | n/a |
| 1 slot - generate | 250k | not run | 25.6 | n/a |
| 5 slots - concurrent - prefill | 5k | not run | 382 (165/slot) | n/a |
| 5 slots - concurrent - generate | 5k | not run | 149.7 (35.4/slot) | n/a |
| 5 slots - concurrent - prefill | 50k | not run | 331 (246/slot) | n/a |
| 5 slots - concurrent - generate | 50k | not run | 122.3 (30.5/slot) | n/a |
| 5 slots - concurrent - prefill | 150k | not run | not run | n/a |
| 5 slots - concurrent - generate | 150k | not run | not run | n/a |
| 5 slots - concurrent - prefill | 200k | not run | not run | n/a |
| 5 slots - concurrent - generate | 200k | not run | not run | n/a |
| 5 slots - concurrent - prefill | 250k | not run | not run | n/a |
| 5 slots - concurrent - generate | 250k | not run | not run | n/a |
