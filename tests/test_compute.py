import json

import numpy as np
import pytest

from compute_sanity import check_equivalence, save_results, step_environment
from devices import require_device
from timing import summarize


def sample(seconds, repeat=0, operations=1, items=4):
    return dict(metric="forward", device="cpu", threads=1, width=64, batch=4,
                repeat=repeat, iteration=0, seconds=seconds, operations=operations, items=items)


def test_summary_reduces_repetitions_before_quantiles():
    # Repetition means: 2 and 8 seconds/op, regardless of unequal sample counts.
    summary = summarize([sample(1), sample(3), sample(8, repeat=1)])[0]
    assert summary["median_seconds"] == 5
    assert summary["q1_seconds"] == 3.5
    assert summary["q3_seconds"] == 6.5
    assert summary["median_items_per_second"] == 1.25
    assert summary["repetitions"] == 2


def test_block_throughput_uses_samples_not_batches():
    summary = summarize([sample(2, operations=10, items=40)])[0]
    assert summary["median_seconds"] == 0.2
    assert summary["median_items_per_second"] == 20


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_timings_rejected(seconds):
    with pytest.raises(ValueError):
        summarize([sample(seconds)])


@pytest.mark.parametrize("termination", ["terminated", "truncated"])
def test_environment_resets_and_transition_accounting(termination):
    class Environment:
        def __init__(self):
            self.reset_calls = 0
            self.count = 0

        def reset(self, seed=None):
            self.reset_calls += 1
            self.count = 0
            return np.zeros(3), {}

        def step(self, action):
            self.count += 1
            done = self.count == 2
            return np.zeros(3), 0, done and termination == "terminated", done and termination == "truncated", {}

    env = Environment()
    result = step_environment(env, lambda observation, index: np.zeros(1), 5, 0)
    assert result["transitions"] == 5
    assert result["resets"] == 2
    assert env.reset_calls == 3  # Initial reset is not a timed episode reset.
    assert result["total_seconds"] == pytest.approx(result["step_seconds"] + result["reset_seconds"])


def test_unavailable_device_is_explicit(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        require_device("cuda")
    with pytest.raises(ValueError, match="Unknown"):
        require_device("magic")


def test_seeded_model_and_gradient_reference():
    result = check_equivalence(64, 4, "cpu", seed=0)
    assert result["passed"]
    assert result["max_gradient_abs_error"] == 0


def test_artifact_round_trip(tmp_path):
    rows = [sample(0.01)]
    save_results(tmp_path, rows, [], [])
    assert json.loads((tmp_path / "raw.json").read_text()) == rows
    assert json.loads((tmp_path / "summary.json").read_text())[0]["median_items_per_second"] == 400
    assert "median_seconds" in (tmp_path / "summary.csv").read_text()
