"""Read a predictions file and score every block against a flat metrics dict.

Block format (see docs/PREDICTIONS.md):

    ### H1.pendulum.fp64
    - statement: free text
    - requires: perturbation_rate.pendulum.fp64 >= 0.5
    - check: flip_rate.pendulum.fp64 >= 0.10

Status rules
    NOT RUN        a metric named in a requires or check line is missing
    NOT TESTABLE   a requires line is false (the instrument or regime is invalid)
    HIT            every check line holds
    MISS           otherwise
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
CLAUSE = re.compile(r"^\s*(\S+)\s*(>=|<=|==|>|<)\s*(-?[\d.eE+-]+)\s*$")


@dataclass
class Clause:
    metric: str
    op: str
    value: float

    @classmethod
    def parse(cls, text: str) -> "Clause":
        m = CLAUSE.match(text)
        if not m:
            raise ValueError(f"cannot parse clause: {text!r}")
        return cls(m.group(1), m.group(2), float(m.group(3)))

    def holds(self, metrics: dict) -> bool:
        return bool(OPS[self.op](metrics[self.metric], self.value))

    def __str__(self) -> str:
        return f"{self.metric} {self.op} {self.value:g}"


@dataclass
class Prediction:
    id: str
    fields: dict = field(default_factory=dict)
    requires: list = field(default_factory=list)
    checks: list = field(default_factory=list)


@dataclass
class Score:
    id: str
    status: str
    observed: dict
    prediction: Prediction


def parse_predictions(text: str) -> list[Prediction]:
    preds: list[Prediction] = []
    current: Prediction | None = None
    for line in text.splitlines():
        if line.startswith("### "):
            current = Prediction(id=line[4:].strip())
            preds.append(current)
            continue
        if current is None:
            continue
        m = re.match(r"^- (\w+): (.*)$", line)
        if not m:
            continue
        key, val = m.groups()
        if key == "requires":
            current.requires.append(Clause.parse(val))
        elif key == "check":
            current.checks.append(Clause.parse(val))
        else:
            current.fields[key] = val
    for p in preds:
        if not p.checks:
            raise ValueError(f"prediction {p.id} has no check line")
    return preds


def score(preds: list[Prediction], metrics: dict) -> list[Score]:
    out = []
    for p in preds:
        clauses = p.requires + p.checks
        observed = {c.metric: metrics.get(c.metric) for c in clauses}
        if any(c.metric not in metrics for c in clauses):
            status = "NOT RUN"
        elif not all(c.holds(metrics) for c in p.requires):
            status = "NOT TESTABLE"
        elif all(c.holds(metrics) for c in p.checks):
            status = "HIT"
        else:
            status = "MISS"
        out.append(Score(p.id, status, observed, p))
    return out


def summarise(scores: list[Score]) -> dict:
    counts: dict[str, int] = {}
    for s in scores:
        counts[s.status] = counts.get(s.status, 0) + 1
    return counts


def markdown_table(scores: list[Score]) -> str:
    lines = ["| Prediction | Check | Observed | Status |", "|---|---|---|---|"]
    for s in scores:
        chk = "; ".join(str(c) for c in s.prediction.checks)
        obs = "; ".join(
            f"{k.split('.')[0]} = {'n/a' if v is None else format(v, '.3g')}" for k, v in s.observed.items()
            if k in {c.metric for c in s.prediction.checks}
        )
        lines.append(f"| {s.id} | {chk} | {obs} | {s.status} |")
    return "\n".join(lines)
