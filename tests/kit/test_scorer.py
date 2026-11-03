from pathlib import Path

import pytest

from thabat_kit import scorer

ROOT = Path(__file__).resolve().parents[2]


BLOCK = """
intro text that is ignored
### T1
- statement: something
- requires: gate >= 0.5
- check: rate >= 0.1
- check: other <= 2
### T2
- check: rate > 0.5
"""


def test_scorer_statuses():
    preds = scorer.parse_predictions(BLOCK)
    assert [p.id for p in preds] == ["T1", "T2"]
    s = {x.id: x.status for x in scorer.score(preds, {"gate": 0.9, "rate": 0.3, "other": 1})}
    assert s == {"T1": "HIT", "T2": "MISS"}
    s = {x.id: x.status for x in scorer.score(preds, {"gate": 0.1, "rate": 0.3, "other": 1})}
    assert s["T1"] == "NOT TESTABLE"
    s = {x.id: x.status for x in scorer.score(preds, {"gate": 0.9, "rate": 0.3})}
    assert s["T1"] == "NOT RUN"
    s = {x.id: x.status for x in scorer.score(preds, {"gate": 0.9, "rate": 0.3, "other": 5})}
    assert s["T1"] == "MISS"
    sc = scorer.score(preds, {"gate": 0.9, "rate": 0.3, "other": 1})
    assert scorer.summarise(sc) == {"HIT": 1, "MISS": 1}
    assert "| T1 |" in scorer.markdown_table(sc)


def test_scorer_rejects_bad_blocks():
    with pytest.raises(ValueError):
        scorer.parse_predictions("### A\n- statement: no check here\n")
    with pytest.raises(ValueError):
        scorer.Clause.parse("not a clause")


def test_real_predictions_file_parses():
    preds = scorer.parse_predictions((ROOT / "docs" / "PREDICTIONS.md").read_text())
    ids = [p.id for p in preds]
    assert len(ids) == len(set(ids)) and len(ids) >= 30
    for p in preds:
        for key in ("statement", "expected", "falsified_if"):
            assert p.fields.get(key), f"{p.id} lacks {key}"
