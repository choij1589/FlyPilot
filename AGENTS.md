# Experiment workflow

## Code and entry points

- Place reusable Python modules directly under `python/`; do not introduce a nested project package. Keep correctness tests under `tests/`.
- Register every runnable Python task as a named command in root-level `doThis.sh`. Extend its help and README usage as steps progress. Use this CLI for experiment execution; do not add interactive menus or an implicit run-all action.
- Keep notebooks thin: import shared analysis/plotting code and load saved measurements instead of duplicating implementations or automatically running benchmarks.
- Use PyROOT with `cmsstyle` for experiment plots. Label figures as FlyPilot; preserve units, uncertainty, and numerical table alternatives. Export PNG, PDF, and ROOT canvases. Restyling saved results must create new render artifacts without overwriting measurements or earlier reports.
- Use the `flypilot` micromamba environment. Interactive activation and the launcher use Zsh. Preserve macOS/Linux portability and report unavailable devices explicitly.

## Before an experiment

- State the question, hypothesis, expected mechanism, controls, metrics with units, and success criteria before execution.
- Run relevant correctness checks before interpreting performance. Separate correctness, timing measurements, and scientific conclusions.
- Fix and record seeds, workload sizes, dtype, device, thread count, software versions, timing boundaries, warm-up, and repetitions. Compare matched inputs and parameters.
- Synchronize asynchronous accelerators at measurement boundaries. State whether transfers, resets, gradient clearing, and optimizer updates are included.

## Results and documentation

- Preserve configuration, metadata, raw measurements, summaries, and figures in unique `results/<run-id>/` directories. Never silently overwrite a previous run. Keep failed runs identifiable.
- Write one numbered English notebook per research step under `notebooks/`. Include question/hypothesis, relative code links and symbol references, mathematical formulas with symbol meanings, mathematical intuition, protocol, visualizations, measured interpretation, limitations, and next questions.
- Document the scope of memory measurements and uncertainty. Missing data is not zero; timing repetitions are not independent RL training seeds.
- Render notebooks from a fresh kernel through the CLI. Save the executed notebook and HTML alongside the corresponding run. Inspect figures and errors before delivery.
- Keep measured observations distinct from proposed explanations. Preserve negative and inconclusive results; never manufacture measurements or extrapolate hardware timing into policy quality.
- Update the README experiment index with date, configuration, findings, intuition, and notebook/artifact links. Complete a milestone only after its checks pass; identify untested platforms.
- Record actual reproduction commands. Revisit these rules after user feedback. Stop after the requested research step; do not advance automatically.
