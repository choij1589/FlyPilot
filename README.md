# FlyPilot

## Abstract

FlyPilot explores whether biological connectome topology can serve as a useful structural prior for reinforcement-learning-based autonomous control. The project starts with hands-on implementations of RL algorithms and a simple bicycle-model driving environment, then compares MLP, recurrent, random sparse, and Drosophila connectome policies under controlled conditions. The goal is to understand learning dynamics and measure topology's effect on sample efficiency, robustness, and generalization—not to assume that biological networks outperform conventional architectures.

See [the research and learning plan](docs/Plan.md) for equations, references, and experimental details. The plan uses the working name **FlyDrive**.

## Environment setup

Use the same Python 3.11 environment on Apple Silicon macOS and x86-64 Linux, including AlmaLinux 9 and NERSC. Interactive activation supports **Zsh only**; the installer itself runs with Bash. Installation needs network access, `curl`, and `tar` with bzip2 support, but no administrator privileges.

```sh
# Once per machine (safe to rerun):
bash scripts/first_setup.sh

# In each Zsh session:
source setup.sh
./doThis.sh check-environment --device cpu
```

The installer reuses micromamba if available, otherwise installs micromamba 2.3.2 under `${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/bin`. It creates the `flypilot` environment there with Python 3.11.13 and pip 25.1.1. Shared direct dependencies are pinned in [scripts/requirements.txt](scripts/requirements.txt); PyTorch 2.7.1 is installed separately for the selected platform. Transitive dependencies are not fully locked, and cross-device numerical results need not be identical. Existing environments must use Python 3.11; the ROOT installation reconciles them to the pinned Python 3.11.13 build.

macOS defaults to an MPS-capable build; Linux defaults to CPU. Check Apple GPU access with `./doThis.sh check-environment --device mps`. For NVIDIA GPUs on Linux:

```sh
bash scripts/first_setup.sh --backend cuda
source setup.sh
./doThis.sh check-environment --device cuda
```

CUDA uses the official PyTorch CUDA 12.6 wheel, with a system-provided compatible NVIDIA driver. A separate CUDA toolkit is not required for these prebuilt wheels. The selected build targets the project's RTX 2070, RTX 3090, and A100 machines; verify it on each target. See [PyTorch installation variants](https://pytorch.org/get-started/previous-versions/#v271). To switch back, rerun with `--backend cpu`; switching replaces the torch build but may leave unused CUDA dependencies installed.

For NERSC, choose a persistent software directory you can write to. Export the same root in every new shell or batch job; the installer does not edit `.zshrc` or cluster modules. [NERSC's Python guidance](https://docs.nersc.gov/development/languages/python/nersc-python/) discusses project software storage.

```sh
# Replace the example with your allocated project/user directory:
export MAMBA_ROOT_PREFIX=/global/common/software/YOUR_PROJECT/YOUR_USER/micromamba
bash scripts/first_setup.sh --root-prefix "$MAMBA_ROOT_PREFIX" --backend cuda
source setup.sh
```

Installation checks imports, package consistency, Pendulum dynamics execution, and CPU tensor gradients. It does not require GPU access on a login node. Run the CUDA smoke check **inside a GPU allocation**. For a noninteractive job, the launcher activates its own environment:

```sh
export MAMBA_ROOT_PREFIX=/path/to/your/micromamba
/path/to/FlyPilot/doThis.sh check-environment --device cuda
```

If reusing a micromamba executable installed elsewhere, expose it through `PATH` or `MAMBA_EXE`. Prefer a clean shell without conflicting Python modules or `PYTHONPATH`/`PYTHONHOME` overrides. Neither installation nor activation changes your shell startup files. Linux/AlmaLinux and NERSC runtime validation must be performed on those machines.

## Plotting with ROOT and cmsstyle

Experiment figures use **PyROOT 6.34.10 and cmsstyle 0.5.0**. The installer obtains `root_base` from conda-forge and `cmsstyle` from pip; see the [ROOT installation guidance](https://root.cern.ch/install/) and [cmsstyle documentation](https://cmsstyle.readthedocs.io/). ROOT's supporting libraries are managed by micromamba. Matplotlib remains available for compatibility with older work, but it is no longer the experiment plotting backend.

Run `bash scripts/first_setup.sh` to update another machine, then restart any existing notebook kernel. Linux/NERSC installation and runtime validation remain pending. Adding ROOT changes supporting libraries and the NumPy binary distribution even though the Python/NumPy/PyTorch version pins are maintained; previous timing results describe the environment recorded when they were measured.

`./doThis.sh render-notebook --run-dir results/<run-id>` regenerates plots from the saved summary into a new render directory. Each figure is exported as PNG, vector PDF, and an editable ROOT canvas. Plots use FlyPilot labels, preserve median/IQR values, and share y-axis ranges between matching width panels. A plot metadata file records ROOT/cmsstyle versions, source hash, and input-summary hash. Existing raw data and prior reports remain untouched.

Both experiment reports linked below have been regenerated with ROOT/cmsstyle. Eighteen tests passed, including reopening the exported ROOT canvas to check graph values/error bars and ensuring an existing figure directory cannot be overwritten.

## Running research steps

The installer registers a Jupyter kernel named **Python (flypilot)**. Select it when opening a notebook. For an environment created before kernel registration was added, run `./doThis.sh register-kernel`, then reload the notebook editor. In VS Code, use **Select Kernel → Select Another Kernel → Jupyter Kernel → Python (flypilot)**. If it is still missing, select the Python environment at `${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/flypilot/bin/python`. Run registration again if the environment is moved or recreated at another path; it updates the user-level `flypilot` kernel registration.

[AGENTS.md](AGENTS.md) defines the experiment and documentation rules. Reusable modules live directly in `python/`; all Python tasks are exposed through [doThis.sh](doThis.sh). The launcher works from any directory and forwards CLI arguments without changing your working directory. Relative output paths refer to your current directory; the default results directory is in this project.

```zsh
./doThis.sh --help
./doThis.sh test -q
./doThis.sh compute-sanity --devices cpu mps
./doThis.sh render-notebook --run-dir results/<printed-run-id>
```

Use `compute-sanity --help` for workload controls. On Linux use `--devices cpu` or `--devices cpu cuda` inside a GPU allocation. Each run saves its configuration, machine metadata, source hashes, correctness checks, raw measurements, summary CSV/JSON, memory snapshots, and figures in a unique directory. Failed runs retain a status and partial measurements. Rendering creates a new subdirectory with an executed notebook and HTML; it does not rerun the benchmark or overwrite earlier renders.

[Notebook 01](notebooks/01_compute_sanity.ipynb) is the reusable documentation source. Read the executed notebook or HTML linked in the experiment index for populated outputs. We will add new launcher commands and numbered notebooks as research steps progress.

## Milestones

Roadmap recorded on **2026-09-17**. Dates are intentionally **TBD** because the plan does not specify deadlines. Replace them with `YYYY-MM-DD`; check a milestone when its completion criteria are met.

### Phase I — Physics and RL laboratory

- [x] **Compute sanity check** — Benchmark CPU/MPS environment, forward, and backward timings, throughput, and memory on M1 Max. **Target:** TBD · **Completed:** 2026-09-17
- [ ] **Dynamics and Bellman foundations** — Reproduce Pendulum dynamics; calculate V, Q, and A in a tiny MDP; implement value iteration, policy iteration, and tabular Q-learning. **Target:** TBD · **Completed:** —
- [ ] **DQN** — Reproduce CartPole learning curves and test replay-buffer and target-network ablations. **Target:** TBD · **Completed:** —
- [ ] **DPG / DDPG** — Understand the deterministic policy gradient and implement DDPG on Pendulum. **Target:** TBD · **Completed:** —
- [ ] **PPO** — Implement PPO with an MLP policy and benchmark Pendulum and MountainCarContinuous. **Target:** TBD · **Completed:** —
- [ ] **TD3 / SAC** — Study TD3's critic-bias correction; implement SAC and compare its learning curves and sample efficiency with PPO. **Target:** TBD · **Completed:** —
- [ ] **LaneKeeping-v0** — Implement a Gymnasium bicycle-model environment with fixed-speed steering control; validate deterministic dynamics before adding sensor noise. **Target:** TBD · **Completed:** —
- [ ] **Classical control baseline** — Validate the environment and reward using a feedback steering controller; record tracking error and control effort. **Target:** TBD · **Completed:** —
- [ ] **Baseline protocol and Phase I exit** — Fix seeds and evaluation settings; compare PPO/SAC with reference implementations; preserve MLP/RNN curves; prepare a graph-policy interface. **Target:** TBD · **Completed:** —

### Phase II — Connectome experiments

Begin only after the baseline prerequisites in [the plan](docs/Plan.md) are satisfied.

- [ ] **Small graph policies** — Implement random sparse recurrent and fly-connectome subgraph policies, starting around 1,000 neurons. **Target:** TBD · **Completed:** —
- [ ] **Controlled topology comparison** — Compare MLP, generic RNN, random sparse, rewired, and fly graphs across multiple seeds, controlling capacity, dynamics, optimizer, and training budget as closely as possible. **Target:** TBD · **Completed:** —
- [ ] **Robustness and generalization** — Measure sensor-noise sensitivity, lesion sensitivity, and performance on unseen road geometry. **Target:** TBD · **Completed:** —
- [ ] **Scaling study** — Expand graph sizes as resources allow and test whether observed effects persist. **Target:** TBD · **Completed:** —
- [ ] **Optional extension** — Consider CARLA or vision after the core experiments are complete. **Target:** TBD · **Completed:** —

## Tests, results, and intuitions

Environment smoke checks are recorded below; no RL experiment results have been recorded yet. Use environment interactions as the primary learning-curve axis; retain individual seed curves and report uncertainty for comparisons. A single debugging run is not evidence of an architecture advantage.

### Experiment index

| Date | Test / experiment | Configuration / seeds | Main result | Intuition / next question | Record / artifacts |
| --- | --- | --- | --- | --- | --- |
| 2026-09-17 | Environment setup smoke check | macOS arm64; Python 3.11.13; PyTorch 2.7.1 | Dependency checks, Pendulum reset/step, CPU/MPS gradients, repeated installation and Zsh activation passed | MPS is usable; this does not establish a speed advantage over CPU. Profile next. Linux/CUDA checks remain pending. | [Smoke-check script](python/check_environment.py) |
| 2026-09-17 | EXP-001: Compute sanity | M1 Max 32 GB; CPU 1/8 threads and MPS; seed 0; 5 repetitions | CPU rollout ≈29,787 transitions/s versus MPS ≈1,240; MPS wins the largest batched compute case | Use CPU for initial single-environment interaction; reassess training device with the actual batch and algorithm. | [Executed notebook](results/20260917T031705Z_compute_1f3b691a/render_20260917T042359Z_9da0f6/01_compute_sanity.ipynb) · [HTML report](results/20260917T031705Z_compute_1f3b691a/render_20260917T042359Z_9da0f6/01_compute_sanity.html) · [Summary CSV](results/20260917T031705Z_compute_1f3b691a/summary.csv) |
| 2026-09-17 | EXP-002: Wider compute sweep | Widths 512/1,024; batches 1–4,096; same host/protocol | MPS is 6.41×/6.83× faster than the faster CPU setting at batch 4,096 | Larger dense workloads favor MPS; first winning sampled batches are 256/64 respectively. | [Executed notebook](results/20260917T033507Z_compute_e430bc7a/render_20260917T042335Z_e02931/01_compute_sanity.ipynb) · [Summary CSV](results/20260917T033507Z_compute_e430bc7a/summary.csv) |

### EXP-001 — Findings and interpretation

The expanded-width follow-up is recorded below as EXP-002.

The [saved run](results/20260917T031705Z_compute_1f3b691a/) contains the raw data and reproduction configuration. Seventeen correctness/CLI tests and all sixteen matched CPU/MPS forward-and-gradient checks passed. Five timing repetitions were collected per configuration; these are not independent RL training runs.

| Workload | CPU, 1 thread | CPU, 8 threads | MPS |
| --- | ---: | ---: | ---: |
| Single-environment rollout, transitions/s | 29,787 | 29,476 | 1,240 |
| Width 64, batch 1,024 forward/backward, samples/s | 1,414,585 | 1,394,835 | 1,075,820 |
| Width 256, batch 1,024 forward/backward, samples/s | 349,217 | 623,953 | 996,469 |

Values are medians; the notebook and CSV include interquartile ranges. Environment-only stepping took about 7.81 µs/transition, excluding separately measured episode resets. CPU was approximately 24× faster than MPS for this single-environment rollout. MPS was approximately 1.60× faster than the eight-thread CPU for the largest compute case. More CPU threads did not consistently help the smaller workloads.

**Intuition:** Fixed accelerator dispatch and transfer costs plausibly dominate sequential small-network calls; batching gives computation more opportunity to amortize those costs. This experiment observes the workload dependence but does not isolate each overhead mechanism. Begin single-environment work on CPU and remeasure the eventual training loop before choosing its device.

**Limits:** The network is untrained, the synthetic loss has no optimizer update, and timing depends on this host and software. RSS is a process lifetime peak; MPS counters are allocation snapshots. Linux/CUDA/NERSC validation remains pending. One initial notebook-render attempt was blocked by the execution sandbox's local-socket restriction; its failed notebook is retained, and the linked fresh-kernel render succeeded outside the sandbox.

### EXP-002 — Widths 512/1,024 and batches through 4,096

**Date:** 2026-09-17. **Question:** Does a larger dense workload shift the CPU/MPS crossover? Same host, seed, dtype, timing definitions, warm-up and five repetitions as EXP-001. All 20 matched output/gradient checks passed. The auxiliary rollout remains width 64.

```zsh
./doThis.sh compute-sanity --devices cpu mps --widths 512 1024 --batches 1 64 256 1024 4096
./doThis.sh render-notebook --run-dir results/20260917T033507Z_compute_e430bc7a
```

| Hidden width | First tested batch where MPS beats both CPU settings | CPU time at batch 4,096, faster setting | MPS time at batch 4,096 | MPS speedup |
| --- | ---: | ---: | ---: | ---: |
| 512 | 256 | 13.09 ms (8 threads) | 2.04 ms | 6.41× |
| 1,024 | 64 | 32.48 ms (8 threads) | 4.75 ms | 6.83× |

Times are median forward/backward compute-block time per batch, excluding optimizer updates and input transfers. The first winning batch is the first **sampled** point, not an exact crossover threshold. MPS remains slower at batch 1 for both widths. At the first winning batches, MPS's timing IQR is below both CPU IQRs, though this is descriptive variability rather than a significance test.

**Interpretation:** The small-network conclusion does not generalize to larger dense training workloads. Increasing width makes the MPS advantage appear at smaller batches, consistent with useful computation amortizing dispatch overhead. CPU remains a good starting point for the fixed small sequential rollout; training-device selection should account for actual model width and batch size. Dense MLP performance still does not establish sparse connectome performance.

Read the [executed notebook](results/20260917T033507Z_compute_e430bc7a/render_20260917T042335Z_e02931/01_compute_sanity.ipynb), [HTML report](results/20260917T033507Z_compute_e430bc7a/render_20260917T042335Z_e02931/01_compute_sanity.html), or [summary CSV](results/20260917T033507Z_compute_e430bc7a/summary.csv). Figures preserve the original single-thread CPU speedup baseline; the table above instead compares MPS with the faster measured CPU setting.

### Experiment entry template

Copy this section for each test or experiment.

#### EXP-XXX — Title

- **Date:** YYYY-MM-DD
- **Milestone:**
- **Question / hypothesis:**
- **Intuition before testing:** What behavior do you expect, and why?
- **Setup:** Environment/version, algorithm, policy architecture, hardware/device, seeds, hyperparameters, and environment-interaction budget.
- **Reproduction:** Command or notebook, code revision if available, configuration, and artifact paths.
- **Controls / success criteria:** What is held fixed, what changes, and what would support or contradict the hypothesis?

| Measurement | Result | Baseline / comparison | Uncertainty / seed variation |
| --- | --- | --- | --- |
| Return vs. environment interactions | — | — | — |
| RMS lateral / heading error (driving) | — | — | — |
| Steering effort / smoothness (driving) | — | — | — |
| Failure / completion rate | — | — | — |
| Training time / inference latency / steps per second | — | — | — |
| Peak memory | — | — | — |

Use only applicable measurements; add test-specific checks such as Bellman residuals or dynamics agreement.

- **Observations:** What did the measurements and curves actually show?
- **Intuition after testing:** What mechanism might explain the result? Keep explanations distinct from measured evidence.
- **Limitations / alternative explanations:** Implementation issues, seed variance, capacity differences, or other confounders.
- **Conclusion:** Supported, contradicted, or inconclusive—and why.
- **Next test:** The smallest experiment that would clarify the remaining question.
