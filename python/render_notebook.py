"""Execute the step-01 documentation against a completed, immutable benchmark run."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

REPO_ROOT = Path(__file__).resolve().parents[1]


def render(run_dir):
    run_dir = Path(run_dir).resolve()
    status = json.loads((run_dir / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Only completed benchmark runs can be rendered")
    notebook = nbformat.read(REPO_ROOT / "notebooks/01_compute_sanity.ipynb", as_version=4)
    render_id = datetime.now(timezone.utc).strftime("render_%Y%m%dT%H%M%SZ_") + uuid4().hex[:6]
    output = run_dir / render_id
    output.mkdir(exist_ok=False)
    from plotting import plot_results
    plot_results(run_dir, output / "figures")
    for cell in notebook.cells:
        if "run-parameters" in cell.metadata.get("tags", []):
            cell.source = (f"from pathlib import Path\nRUN_DIR = Path({str(run_dir)!r})\n"
                           f"FIGURE_DIR = Path({str(output / 'figures')!r})")
    try:
        NotebookClient(notebook, timeout=120, kernel_name="flypilot",
                       resources={"metadata": {"path": str(REPO_ROOT)}}).execute()
    except Exception:
        nbformat.write(notebook, output / "failed.ipynb")
        raise
    # Source notebook lives one level below root; rendered notebook lives three below.
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace("../python/", "../../../python/")
    nbformat.write(notebook, output / "01_compute_sanity.ipynb")
    body, _ = HTMLExporter().from_notebook_node(notebook)
    (output / "01_compute_sanity.html").write_text(body)
    print(f"Executed notebook and HTML: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    render(args.run_dir)


if __name__ == "__main__":
    main()
