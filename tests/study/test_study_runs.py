"""Fast checks of the experiment code on very short runs. The outcomes here are not results."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from thabat_kit.env import NumericalEnv  # noqa: E402
from thabat_study import grok  # noqa: E402
from thabat_study.dqn import run_dqn  # noqa: E402
from thabat_study.td3 import run_td3  # noqa: E402
from thabat_study.tasks import ARMS, load_cfg, runner_for  # noqa: E402

REF = NumericalEnv("ref")
F64 = NumericalEnv("fp64", dtype="float64")


def tiny(task):
    cfg = load_cfg(task, need_budget=False)
    if task == "grok":
        cfg.update(steps=30, train_frac=0.5, eval_every=10)
    elif task == "cartpole":
        cfg.update(steps=300, start_steps=50, eval_every=150, eval_episodes=2, eval_episodes_curve=1)
    else:
        cfg.update(steps=200, start_steps=50, eval_every=100, eval_episodes=2, eval_episodes_curve=1)
    return cfg


@pytest.mark.parametrize("task", ["grok", "pendulum", "invpend", "cartpole"])
def test_same_environment_is_bitwise_repeatable_and_fp64_differs(task):
    cfg, fn = tiny(task), runner_for(task)
    REF.apply()
    a, b = fn(3, REF, cfg), fn(3, REF, cfg)
    assert a["param_hash"] == b["param_hash"] and a["score"] == b["score"]
    assert fn(4, REF, cfg)["param_hash"] != a["param_hash"]
    F64.apply()
    try:
        assert fn(3, F64, cfg)["param_hash"] != a["param_hash"]
    finally:
        REF.apply()
    assert set(a) >= {"outcome", "score", "param_hash", "curve"}


def test_init_is_identical_across_dtypes():
    import torch

    from thabat_study.common import float32_default

    with float32_default():
        torch.manual_seed(1)
        m32 = grok.TinyTransformer(23)
    m64 = m32.double()
    with float32_default():
        torch.manual_seed(1)
        again = grok.TinyTransformer(23)
    assert all(torch.equal(p.double(), q.double()) for p, q in zip(m64.parameters(), again.parameters()))


def test_grok_model_size_and_split():
    import numpy as np

    assert 11500 < grok.n_params(23) < 13000
    (xt, _), (xv, _) = grok.make_data(23, 0.5, np.random.default_rng(0))
    assert len(xt) + len(xv) == 23 * 23 and len(xt) == round(0.5 * 529)
    (xt2, _), _ = grok.make_data(23, 0.5, np.random.default_rng(0))
    assert (xt == xt2).all()


def test_configs_load_and_arms_defined():
    for t in ("pendulum", "cartpole", "invpend", "grok"):
        assert load_cfg(t, need_budget=False)["task"] == t
    assert {"ref", "ref_repeat", "thr", "fp64", "isa", "ref_k", "gpu"} <= set(ARMS)
    assert ARMS["thr"].threads == 4 and ARMS["fp64"].dtype == "float64"
