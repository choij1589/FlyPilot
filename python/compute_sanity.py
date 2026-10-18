"""Step 01: matched CPU/accelerator compute and Pendulum rollout measurements."""

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from devices import memory_snapshot, metadata, require_device, synchronize
from timing import summarize, time_operation

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_case(width, batch, device, seed=0):
    """Build on CPU first so every device receives exactly the same seeded data."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, width),
                              nn.Tanh(), nn.Linear(width, 1))
        x = torch.randn(batch, 3)
        target = torch.randn(batch, 1)
    return model.to(device), x.to(device), target.to(device)


def check_equivalence(width, batch, device, seed):
    reference, x, target = make_case(width, batch, "cpu", seed)
    candidate, dx, dt = make_case(width, batch, device, seed)
    expected, actual = reference(x), candidate(dx)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    nn.functional.mse_loss(expected, target).backward()
    nn.functional.mse_loss(actual, dt).backward()
    errors = []
    for p, q in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(q.grad.cpu(), p.grad, rtol=1e-4, atol=1e-5)
        errors.append(float((q.grad.cpu() - p.grad).abs().max()))
    return {"device": device, "width": width, "batch": batch,
            "max_output_abs_error": float((actual.cpu() - expected).detach().abs().max()),
            "max_gradient_abs_error": max(errors), "passed": True}


def network_samples(width, batch, device, threads, repeat, config):
    model, x, target = make_case(width, batch, device, config["seed"])

    def train_compute():
        model.zero_grad(set_to_none=True)
        nn.functional.mse_loss(model(x), target).backward()

    with torch.inference_mode():
        for _ in range(config["warmup"]):
            model(x)
    for _ in range(config["warmup"]):
        train_compute()
    synchronize(device)
    rows = []

    def record(metric, seconds, iteration, operations=1):
        rows.append({"metric": metric, "device": device, "threads": threads,
                     "width": width, "batch": batch, "repeat": repeat, "iteration": iteration,
                     "seconds": seconds, "operations": operations, "items": operations * batch})

    with torch.inference_mode():
        for iteration in range(config["iterations"]):
            record("forward", time_operation(lambda: model(x), device), iteration)
    for iteration in range(config["iterations"]):
        model.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), target)
        record("backward", time_operation(loss.backward, device), iteration)

    def block():
        for _ in range(config["iterations"]):
            train_compute()

    record("forward_backward", time_operation(block, device), 0, config["iterations"])
    memory = dict(device=device, threads=threads, width=width, batch=batch,
                  repeat=repeat, **memory_snapshot(device))
    return rows, memory


def step_environment(env, action_fn, steps, seed):
    """Count transitions and resets. Report reset-only timing as a subset of total."""
    observation, _ = env.reset(seed=seed)  # Initial reset excluded from timing.
    resets, reset_seconds = 0, 0.0
    start = perf_counter_ns()
    for index in range(steps):
        action = action_fn(observation, index)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            reset_start = perf_counter_ns()
            observation, _ = env.reset()
            reset_seconds += (perf_counter_ns() - reset_start) / 1e9
            resets += 1
    seconds = (perf_counter_ns() - start) / 1e9
    return {"total_seconds": seconds, "step_seconds": seconds - reset_seconds,
            "reset_seconds": reset_seconds, "resets": resets, "transitions": steps}


def environment_samples(device, threads, repeat, config, policy=False):
    env = gym.make("Pendulum-v1")
    if policy:
        model, _, _ = make_case(64, 1, device, config["seed"])

        def action_fn(observation, _):
            x = torch.as_tensor(observation, dtype=torch.float32).reshape(1, 3).to(device)
            # Pendulum torque bounds are [-2, 2]. This is an untrained bounded policy.
            return (2 * torch.tanh(model(x))).reshape(1).cpu().numpy()
    else:
        actions = np.random.default_rng(config["seed"]).uniform(
            -2, 2, size=(config["env_steps"], 1)).astype(np.float32)

        def action_fn(_, index):
            return actions[index % len(actions)]

    try:
        with torch.inference_mode():
            step_environment(env, action_fn, config["warmup"], config["seed"])
            synchronize(device)
            result = step_environment(env, action_fn, config["env_steps"], config["seed"])
            synchronize(device)
    finally:
        env.close()
    seconds = result["total_seconds"] if policy else result["step_seconds"]
    row = {"metric": "rollout" if policy else "environment", "device": device,
           "threads": threads, "width": 64 if policy else 0, "batch": 1,
           "repeat": repeat, "iteration": 0, "seconds": seconds,
           "operations": config["env_steps"], "items": config["env_steps"]}
    return row, dict(device=device, threads=threads, policy=policy, repeat=repeat, **result)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_results(run_dir, rows, memory, resets):
    write_json(run_dir / "raw.json", rows)
    summary = summarize(rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "memory.json", memory)
    write_json(run_dir / "environment_details.json", resets)
    with (run_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def run(config):
    for device in config["devices"]:
        require_device(device)
    if "mps" in config["devices"] and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable PYTORCH_ENABLE_MPS_FALLBACK to avoid silently timing CPU fallback.")
    meta = metadata()
    meta["invocation"] = shlex.join(["./doThis.sh", "compute-sanity", *sys.argv[1:]])
    meta["source_sha256"] = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((REPO_ROOT / "python").glob("*.py"))}
    original_threads = torch.get_num_threads()
    config["cpu_thread_counts"] = sorted({1, original_threads})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_compute_" + uuid4().hex[:8]
    run_dir = Path(config["output_root"]).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metadata.json", meta)
    write_json(run_dir / "status.json", {"status": "running"})
    print(f"Run directory: {run_dir}", flush=True)
    rows, memory, resets, checks = [], [], [], []
    variants = [(d, t) for d in config["devices"]
                for t in (config["cpu_thread_counts"] if d == "cpu" else [1])]
    try:
        torch.set_num_threads(1)
        for device in config["devices"]:
            for width in config["widths"]:
                for batch in config["batches"]:
                    checks.append(check_equivalence(width, batch, device, config["seed"]))
        write_json(run_dir / "correctness.json", checks)
        print("Matched forward/gradient checks passed.", flush=True)
        for repeat in range(config["repetitions"]):
            order = variants if repeat % 2 == 0 else variants[::-1]
            for device, threads in order:
                torch.set_num_threads(threads)
                for width in config["widths"]:
                    for batch in config["batches"]:
                        samples, snapshot = network_samples(width, batch, device, threads, repeat, config)
                        rows.extend(samples)
                        memory.append(snapshot)
                sample, details = environment_samples(device, threads, repeat, config, policy=True)
                rows.append(sample)
                resets.append(details)
                print(f"Repetition {repeat + 1}/{config['repetitions']}: {device}, threads={threads} done", flush=True)
            torch.set_num_threads(1)
            sample, details = environment_samples("cpu", 1, repeat, config)
            rows.append(sample)
            resets.append(details)
        save_results(run_dir, rows, memory, resets)
        from plotting import plot_results
        plot_results(run_dir)
        write_json(run_dir / "status.json", {"status": "complete"})
        print(f"Saved raw measurements, summaries, and figures: {run_dir}", flush=True)
    except Exception as error:
        write_json(run_dir / "raw_partial.json", rows)
        write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        torch.set_num_threads(original_threads)
    return run_dir


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps", "cuda"), default=["cpu", "mps"])
    parser.add_argument("--widths", nargs="+", type=positive_int, default=[64, 256])
    parser.add_argument("--batches", nargs="+", type=positive_int, default=[1, 64, 256, 1024])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--repetitions", type=positive_int, default=5)
    parser.add_argument("--env-steps", type=positive_int, default=1000)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices):
        parser.error("Devices must not be repeated")
    if "cpu" not in args.devices:
        parser.error("Include cpu as the reference device")
    try:
        run(vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
