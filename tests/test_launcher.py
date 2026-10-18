import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    shutil.copy2(ROOT / "doThis.sh", root / "doThis.sh")
    return root


def invoke(root, *args, **kwargs):
    return subprocess.run(["zsh", str(root / "doThis.sh"), *args], cwd=root.parent,
                          capture_output=True, text=True, **kwargs)


def test_help_does_not_require_environment(launcher):
    for args in [(), ("--help",)]:
        result = invoke(launcher, *args)
        assert result.returncode == 0
        assert "compute-sanity" in result.stdout


def test_unknown_command_fails_without_activation(launcher):
    result = invoke(launcher, "unknown")
    assert result.returncode == 2
    assert "Unknown command" in result.stderr


@pytest.mark.parametrize("task,entry", [
    ("compute-sanity", "compute_sanity.py"),
    ("check-environment", "check_environment.py"),
    ("render-notebook", "render_notebook.py"),
    ("test", None),
])
def test_dispatch_preserves_arguments_paths_and_status(launcher, task, entry):
    binary = launcher / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit "${LAUNCHER_TEST_STATUS:-0}"\n')
    python.chmod(0o755)
    (launcher / "setup.sh").write_text(f'export PATH="{binary}:$PATH"\n')
    env = dict(os.environ, LAUNCHER_TEST_STATUS="7")
    result = invoke(launcher, task, "--custom", "value with spaces", env=env)
    assert result.returncode == 7
    arguments = result.stdout.splitlines()
    assert arguments[-2:] == ["--custom", "value with spaces"]
    if entry:
        assert arguments[0] == str(launcher / "python" / entry)
    else:
        assert arguments[:3] == ["-m", "pytest", str(launcher / "tests")]
