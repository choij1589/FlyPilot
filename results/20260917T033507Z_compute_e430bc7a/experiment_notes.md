# EXP-002 — Wider MLP compute sweep

Question: Does increasing hidden width and batch size move the CPU/MPS performance crossover?

Hypothesis stated before measurement: Larger matrix operations should amortize fixed accelerator dispatch costs, making MPS more competitive. This is an expectation, not a measured conclusion.

Command: `./doThis.sh compute-sanity --devices cpu mps --widths 512 1024 --batches 1 64 256 1024 4096`

Controls: Same float32 implementation, seed 0, one/eight CPU threads and MPS, 20 warm-up iterations, 100 measured iterations per repetition, and five repetitions as EXP-001. Forward and gradient agreement must pass before timing. The auxiliary rollout remains width 64; this run expands the synthetic compute sweep, not the rollout policy.

Widths 512 and 1,024 have 265,217 and 1,054,721 parameters respectively. Preserve raw measurements, interquartile ranges, and device allocation snapshots. Compare forward/backward compute-block time against both CPU thread settings; do not interpret synchronized inference latency as training throughput.

## Observations after execution

All 20 matched output/gradient checks passed. The largest output error was 2.16e-7 and largest gradient error was 5.96e-7, below the specified float32 tolerances.

MPS first beats both CPU thread settings at tested batch 256 for width 512 and tested batch 64 for width 1,024. Its IQR is below both CPU IQRs at those points. These sampled points bound the crossover; they do not identify an exact threshold.

At batch 4,096, width 512 takes 13.09 ms on the faster CPU setting (8 threads) and 2.04 ms on MPS, a 6.41× MPS speedup. Width 1,024 takes 32.48 ms on the faster CPU setting (8 threads) and 4.75 ms on MPS, a 6.83× MPS speedup. These are median times per forward/backward batch from block timing, without optimizer updates or input transfers.

Interpretation: Wider dense networks amortize accelerator overhead sooner, consistent with the original hypothesis. CPU wins the batch-1 cases, while MPS wins large batched compute. Do not extend this conclusion to sparse recurrent connectomes without measuring their actual implementation. The reference width-64 rollout still favors CPU and is not a test of large-policy rollout latency.
