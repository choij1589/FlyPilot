import json

import pytest

from root_plotting import plot_results


def test_root_exports_preserve_summary_values_and_old_figures(tmp_path):
    import ROOT

    rows = []
    for metric in ("forward", "backward", "forward_backward", "rollout", "environment"):
        for batch in ([1, 64] if metric not in ("rollout", "environment") else [1]):
            rows.append(dict(metric=metric, device="cpu", threads=1, width=64, batch=batch,
                             median_seconds=0.01, q1_seconds=0.008, q3_seconds=0.012,
                             median_items_per_second=100 * batch,
                             q1_items_per_second=80 * batch, q3_items_per_second=120 * batch))
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(rows))
    before = source.read_bytes()
    output = tmp_path / "figures"
    plot_results(tmp_path, output)
    assert source.read_bytes() == before
    for name in ("latency", "throughput", "speedup", "rollout"):
        for suffix in ("png", "pdf", "root"):
            assert (output / f"{name}.{suffix}").stat().st_size > 100
    file = ROOT.TFile.Open(str(output / "throughput.root"))
    try:
        assert not file.IsZombie()
        canvas = file.Get(file.GetListOfKeys().At(0).GetName())
        pad = next(obj for obj in canvas.GetListOfPrimitives() if obj.InheritsFrom("TPad"))
        graph = next(obj for obj in pad.GetListOfPrimitives() if obj.InheritsFrom("TGraphAsymmErrors"))
        assert graph.GetN() == 2
        assert graph.GetPointX(1) == 64
        assert graph.GetPointY(1) == 6400
        assert graph.GetErrorYlow(1) == 1280
        assert graph.GetErrorYhigh(1) == 1280
        assert any(obj.InheritsFrom("TLatex") and obj.GetTitle() == "FlyPilot"
                   for obj in pad.GetListOfPrimitives())
    finally:
        file.Close()
    with pytest.raises(FileExistsError):
        plot_results(tmp_path, output)
