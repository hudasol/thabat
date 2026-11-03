"""Positive control: modular addition with a small transformer, full batch AdamW.

The seed fixes the weight initialisation and the train and held out split.
Training is full batch with no dropout, so the only thing that differs between
numerical environments is the arithmetic.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from thabat_kit.fingerprint import hash_state_dict
from thabat_kit.seeding import seed_everything

from .common import float32_default, torch_device, torch_dtype


class TinyTransformer(nn.Module):
    def __init__(self, p: int, d: int = 32, heads: int = 4, d_mlp: int = 96):
        super().__init__()
        self.tok = nn.Embedding(p + 1, d)
        self.pos = nn.Parameter(torch.randn(3, d) * 0.02)
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d_mlp), nn.GELU(), nn.Linear(d_mlp, d))
        self.lnf = nn.LayerNorm(d)
        self.out = nn.Linear(d, p)

    def forward(self, x):
        h = self.tok(x) + self.pos
        a = self.ln1(h)
        h = h + self.attn(a, a, a, need_weights=False)[0]
        h = h + self.mlp(self.ln2(h))
        return self.out(self.lnf(h)[:, -1])


def n_params(p: int) -> int:
    return sum(t.numel() for t in TinyTransformer(p).parameters())


def make_data(p: int, train_frac: float, rng: np.random.Generator):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    x = np.stack([a.ravel(), b.ravel(), np.full(p * p, p)], axis=1)
    y = (a.ravel() + b.ravel()) % p
    perm = rng.permutation(p * p)
    n_train = int(round(train_frac * p * p))
    tr, va = perm[:n_train], perm[n_train:]
    return (x[tr], y[tr]), (x[va], y[va])


def run_grok(seed: int, env, cfg: dict) -> dict:
    """One training run. cfg: p, train_frac, steps, lr, wd, threshold, eval_every."""
    bundle = seed_everything(seed)
    p = cfg["p"]
    (xt, yt), (xv, yv) = make_data(p, cfg["train_frac"], bundle.rng("split"))
    dev, dt = torch_device(env), torch_dtype(env)
    with float32_default():
        torch.manual_seed(bundle.int_seed("init"))
        model = TinyTransformer(p)
    model = model.to(device=dev, dtype=dt)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    xt_t, yt_t = torch.as_tensor(xt, device=dev), torch.as_tensor(yt, device=dev)
    xv_t, yv_t = torch.as_tensor(xv, device=dev), torch.as_tensor(yv, device=dev)
    lossf = nn.CrossEntropyLoss()

    def val_acc() -> float:
        model.eval()
        with torch.no_grad():
            return float((model(xv_t).argmax(-1) == yv_t).double().mean().item())

    curve = []
    for step in range(1, cfg["steps"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        lossf(model(xt_t), yt_t).backward()
        opt.step()
        if step % cfg["eval_every"] == 0:
            curve.append([step, round(val_acc(), 4)])
    acc = val_acc()
    return {
        "outcome": int(acc >= cfg["threshold"]),
        "score": acc,
        "param_hash": hash_state_dict(model.state_dict()),
        "curve": curve,
    }
