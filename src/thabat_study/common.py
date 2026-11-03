"""Helpers shared by the experiments."""
from __future__ import annotations

import contextlib

import torch
from torch import nn


@contextlib.contextmanager
def float32_default():
    """Build modules with float32 draws whatever the arm's dtype is, then cast.

    Random initial weights drawn in float64 differ from those drawn in float32,
    which would make the dtype arm a different seed. Drawing in float32 and
    casting keeps the initial point identical up to the cast.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def mlp(n_in: int, n_out: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def torch_dtype(env) -> torch.dtype:
    return torch.float64 if env.dtype == "float64" else torch.float32


def torch_device(env) -> torch.device:
    return torch.device(env.device)
