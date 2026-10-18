"""Batch-mode PyROOT/cmsstyle figures from saved benchmark summaries."""

from array import array
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4


def plot_results(run_dir, output_dir=None):
    import ROOT as rt
    import cmsstyle as cms

    rt.gROOT.SetBatch(True)
    cms.setCMSStyle()
    rt.gStyle.SetOptStat(0)
    run_dir = Path(run_dir)
    output = Path(output_dir) if output_dir else run_dir / "figures"
    output.mkdir(parents=True, exist_ok=False)
    source = run_dir / "summary.json"
    rows = json.loads(source.read_text())
    widths = sorted({r["width"] for r in rows if r["metric"] == "forward"})
    variants = sorted({(r["device"], r["threads"]) for r in rows if r["metric"] == "forward"})
    colors = [rt.TColor.GetColor(c) for c in ("#0072B2", "#D55E00", "#009E73", "#CC79A7")]
    keep = []  # PyROOT objects must outlive canvas serialization.

    def label(variant):
        device, threads = variant
        return f"CPU ({threads} thread{'s' if threads != 1 else ''})" if device == "cpu" else device.upper()

    def text(x, y, value, size=0.035, font=42):
        obj = rt.TLatex()
        obj.SetNDC(True)
        obj.SetTextFont(font)
        obj.SetTextSize(size)
        obj.DrawLatex(x, y, value)
        keep.append(obj)

    def graph(x, y, lo, hi, style_index):
        zero = array("d", [0.] * len(x))
        obj = rt.TGraphAsymmErrors(len(x), array("d", x), array("d", y), zero, zero,
                                  array("d", lo), array("d", hi))
        obj.SetName("graph_" + uuid4().hex)
        obj.SetLineColor(colors[style_index % len(colors)])
        obj.SetMarkerColor(colors[style_index % len(colors)])
        obj.SetMarkerStyle([20, 21, 22, 33][style_index % 4])
        obj.SetLineStyle([1, 2, 7, 3][style_index % 4])
        obj.SetLineWidth(2)
        obj.SetMarkerSize(1.1)
        keep.append(obj)
        return obj

    def decorate(pad, title, note):
        pad.SetLeftMargin(0.17)
        pad.SetRightMargin(0.04)
        pad.SetTopMargin(0.26)
        pad.SetBottomMargin(0.19)

    def stamp(title, note):
        text(0.17, 0.94, "FlyPilot", 0.045, 62)
        text(0.38, 0.94, title, 0.036)
        text(0.17, 0.055, note, 0.027)
        text(0.17, 0.015, run_dir.name, 0.022)

    def axes(frame, x_title, y_title):
        for axis in (frame.GetXaxis(), frame.GetYaxis()):
            axis.SetLabelSize(0.038)
            axis.SetTitleSize(0.041)
            axis.SetTitleFont(42)
            axis.SetLabelFont(42)
        frame.GetXaxis().SetTitle(x_title)
        frame.GetYaxis().SetTitle(y_title)
        frame.GetXaxis().SetTitleOffset(1.15)
        frame.GetYaxis().SetTitleOffset(1.7)

    def save(canvas, name):
        canvas.Update()
        for suffix in ("png", "pdf", "root"):
            canvas.SaveAs(str(output / f"{name}.{suffix}"))
        canvas.Close()
        keep.clear()

    def series(metric, width, variant, mode):
        selected = sorted((r for r in rows if r["metric"] == metric and r["width"] == width
                           and (r["device"], r["threads"]) == variant), key=lambda r: r["batch"])
        x, y, lo, hi = [], [], [], []
        for row in selected:
            x.append(float(row["batch"]))
            if mode == "speedup":
                base = next(r for r in rows if r["metric"] == metric and r["width"] == width
                            and r["batch"] == row["batch"] and r["device"] == "cpu" and r["threads"] == 1)
                value = base["median_seconds"] / row["median_seconds"]
                lower = upper = value
            elif mode == "throughput":
                value, lower, upper = (row[f"{q}_items_per_second"] for q in ("median", "q1", "q3"))
            else:
                value, lower, upper = (row[f"{q}_seconds"] * 1e6 for q in ("median", "q1", "q3"))
            y.append(value)
            lo.append(value - lower)
            hi.append(upper - value)
        return x, y, lo, hi

    for name, metrics, mode, y_title in (
        ("latency", ["forward", "backward", "forward_backward"], "latency", "Time per operation [#mus]"),
        ("throughput", ["forward_backward"], "throughput", "Samples / second"),
        ("speedup", ["forward_backward"], "speedup", "CPU 1-thread time / variant time"),
    ):
        canvas = rt.TCanvas("canvas_" + uuid4().hex, name, 760 * len(widths), 560 * len(metrics))
        canvas.Divide(len(widths), len(metrics), 0.002, 0.002)
        for row_index, metric in enumerate(metrics):
            # Share the y range between widths within each metric.
            all_series = [series(metric, width, variant, mode) for width in widths for variant in variants]
            ymin = min(y - lo for _, ys, lows, _ in all_series for y, lo in zip(ys, lows)) * 0.65
            ymax = max(y + hi for _, ys, _, highs in all_series for y, hi in zip(ys, highs)) * 1.8
            for col, width in enumerate(widths):
                pad = canvas.cd(row_index * len(widths) + col + 1)
                pad.SetLogx(True)
                pad.SetLogy(True)
                note = "Ratio of medians; reference = CPU 1 thread" if mode == "speedup" else "Median and IQR; no optimizer updates or transfers"
                decorate(pad, f"{metric.replace('_', ' + ')}; width {width}", note)
                xs = [r["batch"] for r in rows if r["metric"] == metric]
                frame = pad.DrawFrame(min(xs) / 1.5, ymin, max(xs) * 1.5, ymax)
                axes(frame, "Batch size [samples]", y_title)
                keep.append(frame)
                legend = rt.TLegend(0.17, 0.79, 0.96, 0.88)
                legend.SetNColumns(len(variants))
                legend.SetTextFont(42)
                legend.SetTextSize(0.032)
                legend.SetBorderSize(0)
                legend.SetFillStyle(0)
                for index, variant in enumerate(variants):
                    if mode == "speedup" and variant == ("cpu", 1):
                        line = rt.TLine(min(xs) / 1.5, 1, max(xs) * 1.5, 1)
                        line.SetLineStyle(3)
                        line.SetLineColor(rt.kGray + 2)
                        line.Draw()
                        legend.AddEntry(line, "CPU 1-thread", "l")
                        keep.append(line)
                        continue
                    obj = graph(*series(metric, width, variant, mode), index)
                    obj.Draw("LPZ SAME")
                    legend.AddEntry(obj, label(variant), "lp")
                legend.Draw()
                keep.append(legend)
                pad.RedrawAxis()
                stamp(f"{metric.replace('_', ' + ')}; width {width}", note)
        save(canvas, name)

    environment = [r for r in rows if r["metric"] in ("environment", "rollout")]
    canvas = rt.TCanvas("canvas_" + uuid4().hex, "rollout", 1200, 650)
    decorate(canvas, "Single-environment throughput (width 64 policy)", "Median and IQR; rollout includes transfers and episode resets")
    hist = rt.TH1F("frame_" + uuid4().hex, "", len(environment), 0.5, len(environment) + 0.5)
    hist.SetDirectory(0)
    hist.SetMinimum(0)
    hist.SetMaximum(max(r["q3_items_per_second"] for r in environment) * 1.3)
    axes(hist, "", "Transitions / second")
    hist.GetXaxis().SetLabelSize(0.037)
    hist.GetYaxis().SetTitleOffset(1.3)
    for i, row in enumerate(environment, 1):
        title = "#splitline{Environment only}{resets excluded}" if row["metric"] == "environment" else f"#splitline{{Rollout}}{{{label((row['device'], row['threads']))}}}"
        hist.GetXaxis().SetBinLabel(i, title)
    hist.Draw("AXIS")
    keep.append(hist)
    for i, row in enumerate(environment, 1):
        value = row["median_items_per_second"]
        box = rt.TBox(i - 0.3, 0, i + 0.3, value)
        box.SetFillColor(colors[0])
        box.Draw()
        keep.append(box)
        obj = graph([float(i)], [value], [value - row["q1_items_per_second"]],
                    [row["q3_items_per_second"] - value], 0)
        obj.SetLineColor(rt.kBlack)
        obj.Draw("PZ SAME")
        annotation = rt.TLatex(i, row["q3_items_per_second"] + hist.GetMaximum() * 0.035, f"{value:,.0f}")
        annotation.SetTextAlign(22)
        annotation.SetTextFont(42)
        annotation.SetTextSize(0.032)
        annotation.Draw()
        keep.append(annotation)
    canvas.RedrawAxis()
    stamp("Single-environment throughput (width 64 policy)",
          "Median and IQR; rollout includes transfers and episode resets")
    save(canvas, "rollout")
    (output / "plot_metadata.json").write_text(json.dumps({
        "backend": "PyROOT + cmsstyle", "ROOT": rt.gROOT.GetVersion(), "cmsstyle": version("cmsstyle"),
        "summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return sorted(output.glob("*.png"))
