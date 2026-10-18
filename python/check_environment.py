"""Headless smoke check, exposed by doThis.sh check-environment."""

import argparse
import platform

import gymnasium as gym
import ipykernel
import jupyterlab
import matplotlib
import numpy as np
import pytest
import scipy
import torch
import ROOT
import cmsstyle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--expected-backend", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    print(f"Python {platform.python_version()}, PyTorch {torch.__version__}")
    ROOT.gROOT.SetBatch(True)
    cmsstyle.setCMSStyle()
    print(f"ROOT {ROOT.gROOT.GetVersion()}, cmsstyle import/style check passed")
    print(f"CUDA build: {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}; "
          f"MPS available: {torch.backends.mps.is_available()}")
    if args.expected_backend == "cuda":
        assert torch.version.cuda == "12.6", "Expected the CUDA 12.6 PyTorch build"
    elif args.expected_backend == "mps":
        assert torch.backends.mps.is_built(), "Expected an MPS-capable build"
    elif args.expected_backend == "cpu":
        assert torch.version.cuda is None, "Expected a build without CUDA"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run inside a GPU allocation and check the NVIDIA driver.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable on this host/session.")
    x = torch.tensor([1.0, 2.0, 3.0], device=args.device, requires_grad=True)
    x.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    env = gym.make("Pendulum-v1")
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        observation, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert env.observation_space.contains(observation) and np.isfinite(reward)
    finally:
        env.close()
    print(f"PASS: dependency imports, Pendulum reset/step, and {args.device} forward/backward")


if __name__ == "__main__":
    main()
