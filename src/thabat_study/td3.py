"""A compact TD3 for low dimensional control tasks.

Every random draw comes from a numpy stream derived from the seed, so no torch
random number generator is consumed. This keeps the draws identical across
float32, float64, CPU and GPU. Only network arithmetic differs.
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


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, scale, hidden):
        super().__init__()
        self.net = mlp(n_obs, n_act, hidden)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.scale


class Critic(nn.Module):
    def __init__(self, n_obs, n_act, hidden):
        super().__init__()
        self.q1 = mlp(n_obs + n_act, 1, hidden)
        self.q2 = mlp(n_obs + n_act, 1, hidden)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


def evaluate(env_id: str, policy, n_episodes: int, seed0: int, dt, dev) -> float:
    env = gym.make(env_id)
    total = 0.0
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + k)
        done = False
        while not done:
            with torch.no_grad():
                a = policy(torch.as_tensor(obs, dtype=dt, device=dev)).cpu().numpy()
            obs, r, term, trunc, _ = env.step(a.astype(env.action_space.dtype))
            total += r
            done = term or trunc
    env.close()
    return total / n_episodes


def run_td3(seed: int, env, cfg: dict) -> dict:
    bundle = seed_everything(seed)
    dev, dt = torch_device(env), torch_dtype(env)
    genv = gym.make(cfg["env_id"])
    n_obs, n_act = genv.observation_space.shape[0], genv.action_space.shape[0]
    high = genv.action_space.high
    with float32_default():
        torch.manual_seed(bundle.int_seed("init"))
        actor = Actor(n_obs, n_act, high, cfg["hidden"])
        critic = Critic(n_obs, n_act, cfg["hidden"])
    actor, critic = actor.to(device=dev, dtype=dt), critic.to(device=dev, dtype=dt)
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])
    c_opt = torch.optim.Adam(critic.parameters(), lr=cfg["lr"])

    r_explore, r_start = bundle.rng("explore"), bundle.rng("random_actions")
    r_replay, r_target, r_reset = bundle.rng("replay"), bundle.rng("target_noise"), bundle.rng("env_reset")
    steps = cfg["steps"]
    S = np.zeros((steps, n_obs))
    A = np.zeros((steps, n_act))
    R = np.zeros(steps)
    S2 = np.zeros((steps, n_obs))
    D = np.zeros(steps)
    obs, _ = genv.reset(seed=int(r_reset.integers(2**31)))
    curve, train_returns, ep_ret = [], [], 0.0
    T = lambda x: torch.as_tensor(x, dtype=dt, device=dev)  # noqa: E731

    for t in range(steps):
        noise = r_explore.normal(0, cfg["expl_noise"] * high, size=n_act)
        rand_a = r_start.uniform(-high, high, size=n_act)
        if t < cfg["start_steps"]:
            act = rand_a
        else:
            with torch.no_grad():
                act = actor(T(obs)).cpu().numpy().astype(np.float64)
            act = np.clip(act + noise, -high, high)
        obs2, r, term, trunc, _ = genv.step(act.astype(genv.action_space.dtype))
        S[t], A[t], R[t], S2[t], D[t] = obs, act, r, obs2, float(term)
        ep_ret += r
        obs = obs2
        if term or trunc:
            train_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = genv.reset(seed=int(r_reset.integers(2**31)))
        if t >= cfg["start_steps"]:
            idx = r_replay.integers(0, t + 1, size=cfg["batch"])
            tn = np.clip(r_target.normal(0, cfg["policy_noise"] * high, size=(cfg["batch"], n_act)),
                         -cfg["noise_clip"] * high, cfg["noise_clip"] * high)
            s, a, rr, s2, d = T(S[idx]), T(A[idx]), T(R[idx]).unsqueeze(1), T(S2[idx]), T(D[idx]).unsqueeze(1)
            with torch.no_grad():
                a2 = torch.max(torch.min(actor_t(s2) + T(tn), T(high)), T(-high))
                q1t, q2t = critic_t(s2, a2)
                y = rr + cfg["gamma"] * (1 - d) * torch.min(q1t, q2t)
            q1, q2 = critic(s, a)
            c_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
            c_opt.zero_grad(set_to_none=True)
            c_loss.backward()
            c_opt.step()
            if t % cfg["policy_delay"] == 0:
                a_loss = -critic(s, actor(s))[0].mean()
                a_opt.zero_grad(set_to_none=True)
                a_loss.backward()
                a_opt.step()
                with torch.no_grad():
                    for net, tgt in ((actor, actor_t), (critic, critic_t)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(1 - cfg["tau"]).add_(p, alpha=cfg["tau"])
        if (t + 1) % cfg["eval_every"] == 0 and (t + 1) < steps:
            curve.append([t + 1, round(evaluate(cfg["env_id"], actor, cfg["eval_episodes_curve"], 900000, dt, dev), 2)])
    genv.close()
    score = evaluate(cfg["env_id"], actor, cfg["eval_episodes"], 100000, dt, dev)
    curve.append([steps, round(score, 2)])
    return {
        "outcome": int(score >= cfg["threshold"]),
        "score": score,
        "param_hash": hash_state_dict({**{f"a{k}": v for k, v in actor.state_dict().items()},
                                       **{f"c{k}": v for k, v in critic.state_dict().items()}}),
        "curve": curve,
    }
