import numpy as np
import pytest

from thabat_kit.env import NumericalEnv, capture_env, env_digest
from thabat_kit.fingerprint import hash_arrays, hash_state_dict
from thabat_kit.seeding import derive_seed, seed_everything


def test_env_validates():
    with pytest.raises(ValueError):
        NumericalEnv("x", dtype="float16")
    with pytest.raises(ValueError):
        NumericalEnv("x", threads=0)
    with pytest.raises(ValueError):
        NumericalEnv("x", device="tpu")


def test_capture_has_expected_keys_and_digest_tracks_threads():
    torch = pytest.importorskip("torch")
    keep = torch.get_num_threads()
    try:
        NumericalEnv("a", threads=1).apply()
        c1 = capture_env(NumericalEnv("a", threads=1))
        NumericalEnv("b", threads=2).apply()
        c2 = capture_env(NumericalEnv("b", threads=2))
    finally:
        torch.set_num_threads(keep)
    for key in ("python", "libraries", "blas", "torch", "env_vars", "requested"):
        assert key in c1
    assert c1["torch"]["threads"] == 1 and c2["torch"]["threads"] == 2
    assert env_digest(c1) != env_digest(c2)
    assert env_digest(c1) == env_digest(dict(c1))


def test_apply_sets_dtype_and_restores():
    torch = pytest.importorskip("torch")
    before = torch.get_default_dtype()
    try:
        NumericalEnv("d", dtype="float64").apply()
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(before)


def test_seed_everything_is_repeatable_and_streams_differ():
    b = seed_everything(5)
    x1 = b.rng("init").normal(size=4)
    x2 = seed_everything(5).rng("init").normal(size=4)
    y = b.rng("noise").normal(size=4)
    assert np.array_equal(x1, x2)
    assert not np.array_equal(x1, y)
    assert derive_seed(5, "a") != derive_seed(6, "a")
    assert derive_seed(5, "a") == derive_seed(5, "a")


def test_seed_everything_seeds_numpy_and_torch():
    torch = pytest.importorskip("torch")
    seed_everything(11)
    a = (np.random.rand(), torch.rand(1).item())
    seed_everything(11)
    b = (np.random.rand(), torch.rand(1).item())
    assert a == b


def test_hash_arrays_properties():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert hash_arrays([a]) == hash_arrays([a.astype(np.float64)])
    assert hash_arrays([a]) != hash_arrays([a + 1e-6])
    assert hash_arrays([a]) != hash_arrays([a.reshape(3, 2)])


def test_hash_state_dict_order_independent():
    torch = pytest.importorskip("torch")
    sd1 = {"b": torch.ones(2), "a": torch.zeros(2)}
    sd2 = {"a": torch.zeros(2), "b": torch.ones(2)}
    assert hash_state_dict(sd1) == hash_state_dict(sd2)
