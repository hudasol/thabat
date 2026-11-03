"""The prose of the results templates states verdicts. These checks fail if the data no longer support them."""
import json
from pathlib import Path

from thabat_kit import scorer

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results" / "summary.json").read_text())["metrics"]
STATUS = {s.id: s.status for s in scorer.score(scorer.parse_predictions((ROOT / "docs" / "PREDICTIONS.md").read_text()), M)}


def test_h1_verdicts_in_the_text():
    for t in ("pendulum", "cartpole", "invpend"):
        assert STATUS[f"H1.{t}.fp64"] == "HIT" and STATUS[f"H1.{t}.isa"] == "HIT"
        assert STATUS[f"H1.{t}.thr"] == "NOT TESTABLE" and STATUS[f"H1.{t}.gpu"] == "NOT RUN"


def test_h2_verdicts_in_the_text():
    for arm in ("fp64", "isa"):
        assert STATUS[f"H2.pendulum.{arm}"] == "HIT" and STATUS[f"H2.cartpole.{arm}"] == "HIT"
        assert STATUS[f"H2.invpend.{arm}"] == "MISS"
        assert M[f"abs_delta.invpend.{arm}"] > 0.10 and M[f"mcnemar_holm_p.invpend.{arm}"] >= 0.05


def test_h3_verdicts_and_ranges_in_the_text():
    for key in ("pendulum", "cartpole", "invpend", "all"):
        assert STATUS[f"H3.{key}"] == "MISS"
    for key in ("pendulum", "cartpole", "all"):
        assert 0.55 <= M[f"auc_dist.{key}"] < 0.70
    assert M["auc_dist.invpend"] < 0.55


def test_thread_arm_is_bit_identical_for_rl_and_changes_grok():
    for t in ("pendulum", "cartpole", "invpend"):
        assert M[f"perturbation_rate.{t}.thr"] == 0.0
        assert M[f"perturbation_rate.{t}.fp64"] == 1.0 and M[f"perturbation_rate.{t}.isa"] == 1.0
    assert M["perturbation_rate.grok.thr"] == 1.0
    assert STATUS["PC1.grok.thr"] == "HIT" and STATUS["PC2.grok.thr"] == "HIT"
