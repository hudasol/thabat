import numpy as np
import pytest

from thabat_kit import stats


def test_wilson_known_values():
    lo, hi = stats.wilson_interval(0, 30)
    assert lo == 0.0 and hi == pytest.approx(0.1135, abs=2e-3)
    lo, hi = stats.wilson_interval(15, 30)
    assert lo == pytest.approx(1 - hi, abs=1e-9)
    assert np.isnan(stats.wilson_interval(0, 0)[0])


def test_rate_with_ci():
    r, lo, hi = stats.rate_with_ci([1, 0, 1, 1])
    assert r == 0.75 and lo < r < hi


def test_bootstrap_reproducible_and_covers_mean():
    x = np.random.default_rng(0).normal(size=200)
    a = stats.bootstrap_ci(x, seed=1, n_boot=2000)
    b = stats.bootstrap_ci(x, seed=1, n_boot=2000)
    assert a == b and a[0] < x.mean() < a[1]
    assert np.isnan(stats.bootstrap_ci([])[0])


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(3, 0) == pytest.approx(0.25)
    assert stats.mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert stats.mcnemar_exact(21, 28) == pytest.approx(0.3912, abs=1e-3)  # the pairs reported by Ootani


def test_holm():
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert all(adj[k] >= p for k, p in {"a": 0.01, "b": 0.04, "c": 0.03}.items())


def test_auc():
    a, p = stats.auc([3, 4, 5], [0, 1, 2])
    assert a == 1.0 and p < 0.1
    assert stats.auc([1, 1], [1, 1])[0] == 0.5
    assert np.isnan(stats.auc([], [1])[0])
    lo, hi = stats.auc_ci([3, 4, 5, 6], [0, 1, 2, 3.5], n_boot=300)
    assert 0.5 < lo <= hi <= 1.0
    assert np.isnan(stats.auc_ci([], [1])[0])


def make(outcomes, hashes=None):
    return {i: {"outcome": o, **({"param_hash": hashes[i]} if hashes else {})} for i, o in enumerate(outcomes)}


def test_compare_arms_synthetic():
    # SYNTHETIC toy outcomes written by hand for the test, not results
    ref = make([1, 1, 0, 0, 1, 0], ["a", "b", "c", "d", "e", "f"])
    arm = make([1, 0, 0, 1, 1, 0], ["a", "x", "c", "y", "e", "z"])
    out = stats.compare_arms(ref, arm, n_boot=200)
    assert out["n"] == 6 and out["lost"] == 1 and out["gained"] == 1 and out["flips"] == 2
    assert out["flip_rate"] == pytest.approx(2 / 6)
    assert out["abs_delta"] == 0.0 and out["mcnemar_p"] == 1.0
    assert out["perturbation_rate"] == pytest.approx(3 / 6)


def test_compare_arms_without_hashes_or_overlap():
    out = stats.compare_arms(make([1, 0]), make([1, 0]))
    assert "perturbation_rate" not in out and out["flips"] == 0
    empty = stats.compare_arms({}, {})
    assert empty["n"] == 0 and np.isnan(empty["flip_rate"])
