"""A compact DQN for a CartPole variant with seeded Gaussian observation noise.

All random draws come from numpy streams derived from the seed. Per step the
same number of draws is consumed whatever the policy does.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from thabat_kit.fingerprint import hash_state_dict
from thabat_kit.seeding import seed_everything

from .common import float32_default, mlp, torch_device, torch_dtype


def evaluate(cfg: dict, qnet, n_episodes: int, seed0: int, dt, dev) -> float:
    env = gym.make(cfg["env_id"])
    noise = np.random.default_rng(seed0)
    total = 0.0
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + k)
        done = False
        while not done:
            o = obs + noise.normal(0, cfg["obs_noise"], size=obs.shape)
            with torch.no_grad():
                a = int(qnet(torch.as_tensor(o, dtype=dt, device=dev)).argmax().item())
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
    env.close()
    return total / n_episodes


def run_dqn(seed: int, env, cfg: dict) -> dict:
    bundle = seed_everything(seed)
    dev, dt = torch_device(env), torch_dtype(env)
    genv = gym.make(cfg["env_id"])
    n_obs, n_act = genv.observation_space.shape[0], genv.action_space.n
    with float32_default():
        torch.manual_seed(bundle.int_seed("init"))
        q = mlp(n_obs, n_act, cfg["hidden"])
    q = q.to(device=dev, dtype=dt)
    qt = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=cfg["lr"])
    r_eps, r_act = bundle.rng("epsilon"), bundle.rng("random_actions")
    r_replay, r_reset, r_obs = bundle.rng("replay"), bundle.rng("env_reset"), bundle.rng("obs_noise")
    steps = cfg["steps"]
    S = np.zeros((steps, n_obs))
    A = np.zeros(steps, dtype=np.int64)
    R = np.zeros(steps)
    S2 = np.zeros((steps, n_obs))
    D = np.zeros(steps)
    T = lambda x: torch.as_tensor(x, dtype=dt, device=dev)  # noqa: E731
    raw, _ = genv.reset(seed=int(r_reset.integers(2**31)))
    obs = raw + r_obs.normal(0, cfg["obs_noise"], size=n_obs)
    curve = []
    decay = cfg["eps_decay_frac"] * steps
    for t in range(steps):
        eps = max(cfg["eps_end"], 1.0 - (1.0 - cfg["eps_end"]) * t / decay)
        u, ra = r_eps.random(), int(r_act.integers(n_act))
        if u < eps:
            act = ra
        else:
            with torch.no_grad():
                act = int(q(T(obs)).argmax().item())
        raw2, r, term, trunc, _ = genv.step(act)
        obs2 = raw2 + r_obs.normal(0, cfg["obs_noise"], size=n_obs)
        S[t], A[t], R[t], S2[t], D[t] = obs, act, r, obs2, float(term)
        obs = obs2
        if term or trunc:
            raw, _ = genv.reset(seed=int(r_reset.integers(2**31)))
            obs = raw + r_obs.normal(0, cfg["obs_noise"], size=n_obs)
        if t >= cfg["start_steps"]:
            idx = r_replay.integers(0, t + 1, size=cfg["batch"])
            s, a, rr, s2, d = T(S[idx]), torch.as_tensor(A[idx], device=dev), T(R[idx]), T(S2[idx]), T(D[idx])
            with torch.no_grad():
                y = rr + cfg["gamma"] * (1 - d) * qt(s2).max(1).values
            pred = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
        if (t + 1) % cfg["target_every"] == 0:
            qt.load_state_dict(q.state_dict())
        if (t + 1) % cfg["eval_every"] == 0 and (t + 1) < steps:
            curve.append([t + 1, round(evaluate(cfg, q, cfg["eval_episodes_curve"], 900000, dt, dev), 2)])
    genv.close()
    score = evaluate(cfg, q, cfg["eval_episodes"], 100000, dt, dev)
    curve.append([steps, round(score, 2)])
    return {
        "outcome": int(score >= cfg["threshold"]),
        "score": score,
        "param_hash": hash_state_dict(q.state_dict()),
        "curve": curve,
    }
