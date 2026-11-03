"""Describe, apply and capture the numerical environment of a run.

A NumericalEnv is the thing a study varies on purpose (device, thread count,
dtype, determinism flags). capture_env records what the process actually
looked like, so a result file can be traced back to the libraries and
settings that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata

LIBRARIES = ("numpy", "scipy", "matplotlib", "torch", "gymnasium", "mujoco")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_CBWR", "CUDA_VISIBLE_DEVICES")
DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class NumericalEnv:
    """A named numerical setting. Only the fields below are varied."""

    name: str
    device: str = "cpu"
    threads: int = 1
    dtype: str = "float32"
    tf32: bool = False
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.threads < 1:
            raise ValueError("threads must be at least 1")
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError(f"device must be cpu or cuda, got {self.device!r}")

    def apply(self) -> None:
        """Set the process wide numerical state described by this object."""
        torch = _try_import("torch")
        if torch is None:
            return
        torch.set_num_threads(self.threads)
        torch.set_default_dtype(getattr(torch, self.dtype))
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cudnn.allow_tf32 = self.tf32
        torch.use_deterministic_algorithms(self.deterministic, warn_only=True)

    def as_dict(self) -> dict:
        return asdict(self)


def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


def _version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _blas_name():
    try:
        import numpy

        cfg = numpy.show_config(mode="dicts")
        return cfg["Build Dependencies"]["blas"].get("name")
    except Exception:
        return None


def _torch_state(torch) -> dict:
    state = {
        "version": torch.__version__,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
        "mkl": bool(torch.backends.mkl.is_available()),
        "mkldnn": bool(torch.backends.mkldnn.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_available": bool(torch.cuda.is_available()),
        "parallel_backend": next(
            (ln.strip() for ln in torch.__config__.parallel_info().splitlines() if "parallel backend" in ln),
            None,
        ),
    }
    if state["cuda_available"]:
        state["cuda_version"] = torch.version.cuda
        state["gpu_name"] = torch.cuda.get_device_name(0)
    return state


def capture_env(env: NumericalEnv | None = None) -> dict:
    """Return a JSON serialisable description of the current process."""
    torch = _try_import("torch")
    cap = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "libraries": {name: _version(name) for name in LIBRARIES},
        "blas": _blas_name(),
        "env_vars": {k: os.environ.get(k) for k in THREAD_VARS},
        "requested": env.as_dict() if env is not None else None,
        "torch": _torch_state(torch) if torch is not None else None,
    }
    return cap


def env_digest(capture: dict) -> str:
    """Short stable digest of a capture. Two runs with the same digest used the same stack."""
    stable = {k: v for k, v in capture.items() if k not in ("platform",)}
    blob = json.dumps(stable, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
