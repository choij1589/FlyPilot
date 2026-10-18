"""Plots and readable tables for saved step-01 results; no benchmark execution."""

import html
import json
from pathlib import Path



def load_summary(run_dir):
    return json.loads((Path(run_dir) / "summary.json").read_text())


def label(device, threads):
    return f"CPU ({threads} thread{'s' if threads != 1 else ''})" if device == "cpu" else device.upper()


def plot_results(run_dir, output_dir=None):
    """Render saved summaries with ROOT/cmsstyle, without replacing existing figures."""
    from root_plotting import plot_results as render
    return render(run_dir, output_dir)


def summary_html(run_dir):
    rows = load_summary(run_dir)
    headings = ["Metric", "Device", "Width", "Batch", "Median µs/op", "IQR µs/op", "Items/s"]
    table = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in headings) + "</tr>"]
    for row in rows:
        values = [row["metric"], label(row["device"], row["threads"]), row["width"], row["batch"],
                  f"{row['median_seconds'] * 1e6:.2f}",
                  f"{row['q1_seconds'] * 1e6:.2f}–{row['q3_seconds'] * 1e6:.2f}",
                  f"{row['median_items_per_second']:,.0f}"]
        table.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    return "".join(table) + "</table>"


def interpretation(run_dir):
    rows = load_summary(run_dir)
    rollouts = [r for r in rows if r["metric"] == "rollout"]
    winner = max(rollouts, key=lambda r: r["median_items_per_second"])
    lines = [f"**Observed rollout result:** {label(winner['device'], winner['threads'])} had the highest "
             f"median single-environment throughput, {winner['median_items_per_second']:,.0f} transitions/s."]
    for width in sorted({r["width"] for r in rows if r["metric"] == "forward_backward"}):
        cases = [r for r in rows if r["metric"] == "forward_backward" and r["width"] == width]
        largest = max(r["batch"] for r in cases)
        best = max((r for r in cases if r["batch"] == largest), key=lambda r: r["median_items_per_second"])
        lines.append(f"**Width {width}, batch {largest}:** {label(best['device'], best['threads'])} had the "
                     f"highest median compute throughput, {best['median_items_per_second']:,.0f} samples/s.")
    lines.append("These are rankings of measured medians, not significance tests. Inspect IQRs and raw "
                 "repetitions before treating small differences as repeatable. Dispatch, synchronization, "
                 "threading, and transfers are candidate explanations; this experiment does not isolate "
                 "their individual contributions. Use the rollout result for a provisional single-environment "
                 "device choice and match the actual training batch before choosing a training device.")
    return "\n\n".join(lines)
