"""Timing and aggregation shared by compute experiments."""

from collections import defaultdict
from time import perf_counter_ns

import numpy as np

from devices import synchronize


def time_operation(operation, device="cpu"):
    """Synchronized wall seconds for one completed operation (includes end sync)."""
    synchronize(device)
    start = perf_counter_ns()
    operation()
    synchronize(device)
    return (perf_counter_ns() - start) / 1e9


def summarize(rows):
    groups = defaultdict(list)
    keys = ("metric", "device", "threads", "width", "batch")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for key, values in groups.items():
        # Reduce each repetition first; do not treat its iterations as independent replicates.
        repeats = defaultdict(list)
        for value in values:
            repeats[value["repeat"]].append(value)
        latencies, throughputs = [], []
        for samples in repeats.values():
            seconds = sum(s["seconds"] for s in samples)
            operations = sum(s["operations"] for s in samples)
            if seconds <= 0 or not np.isfinite(seconds) or operations <= 0:
                raise ValueError("Measurements must have finite positive time and operation count")
            latencies.append(seconds / operations)
            throughputs.append(sum(s["items"] for s in samples) / seconds)
        q1, median, q3 = np.quantile(latencies, [0.25, 0.5, 0.75])
        tq1, tmedian, tq3 = np.quantile(throughputs, [0.25, 0.5, 0.75])
        summaries.append(dict(zip(keys, key), median_seconds=float(median),
                              q1_seconds=float(q1), q3_seconds=float(q3),
                              median_items_per_second=float(tmedian),
                              q1_items_per_second=float(tq1), q3_items_per_second=float(tq3),
                              repetitions=len(repeats)))
    return summaries
