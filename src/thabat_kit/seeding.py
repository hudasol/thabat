"""Seed every random source and derive independent named streams."""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

import numpy as np


def derive_seed(seed: int, name: str) -> int:
    """A 32 bit seed that depends on both the base seed and a stream name."""
    ss = np.random.SeedSequence([int(seed), zlib.crc32(name.encode())])
    return int(ss.generate_state(1)[0])


@dataclass(frozen=True)
class SeedBundle:
    """What seed_everything hands back. Use rng(name) for every private stream."""

    seed: int

    def rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng(derive_seed(self.seed, name))

    def int_seed(self, name: str) -> int:
        return derive_seed(self.seed, name)


def seed_everything(seed: int) -> SeedBundle:
    """Seed python, numpy and (if installed) torch on CPU and CUDA.

    Process level hash randomisation (PYTHONHASHSEED) cannot be set after the
    interpreter has started, so it is recorded by capture_env, not changed here.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return SeedBundle(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return SeedBundle(seed)
