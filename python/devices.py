"""Device validation, synchronization, and explicitly scoped memory metadata."""

import os
import platform
import resource
import subprocess
import sys
from importlib.metadata import version

import torch


def require_device(device):
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device: {device}")
    available = (device == "cpu" or
                 (device == "mps" and torch.backends.mps.is_available()) or
                 (device == "cuda" and torch.cuda.is_available()))
    if not available:
        raise ValueError(f"Requested device {device} is unavailable; use an appropriate host/GPU allocation.")


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory_snapshot(device):
    """RSS is a lifetime high-water mark; MPS counters are current snapshots."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_lifetime_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}
    if device == "mps":
        result.update(mps_tensor_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    elif device == "cuda":
        result.update(cuda_tensor_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved())
    return result


def metadata():
    def sysctl(key):
        if sys.platform != "darwin":
            return None
        try:
            return subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "physical_memory_bytes": sysctl("hw.memsize"), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "executable": sys.executable,
        "versions": {name: version(name) for name in ("torch", "numpy", "gymnasium", "matplotlib")},
        "initial_cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "mps_available": torch.backends.mps.is_available(), "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "memory_note": "RSS is process lifetime peak, not per-case peak. Device counters are snapshots, not peaks.",
    }
