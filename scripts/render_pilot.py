"""Write docs/PILOT.md from the pilot result files (pilot seeds 1000 to 1009 only)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows_of(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def table(paths, score_fmt):
    out = ["| File | Seeds | Success rate | Median score | Min | Max |", "|---|---|---|---|---|---|"]
    for p in paths:
        r = rows_of(p)
        sc = np.array([x["score"] for x in r])
        out.append(f"| {p.stem} | {len(r)} | {np.mean([x['outcome'] for x in r]):.2f} | {format(np.median(sc), score_fmt)} | "
                   f"{format(sc.min(), score_fmt)} | {format(sc.max(), score_fmt)} |")
    return "\n".join(out)


def main() -> None:
    pilot = ROOT / "results" / "pilot"
    fmt = {"grok": ".3f", "pendulum": ".0f", "cartpole": ".0f", "invpend": ".0f"}
    lines = ["# Pilot", "",
             "Author: Huda Mueen. Pilot seeds 1000 to 1009 only, reference environment only. Evaluation seeds are never used here. "
             "The rule and the grid are in configs/pilot_grid.json and docs/PREDICTIONS.md, and every change to them is in the amendment log. "
             "This file is written by scripts/render_pilot.py from the files in results/pilot/.", ""]
    for task in ("pendulum", "cartpole", "invpend", "grok"):
        lines += [f"## {task}", ""]
        summ = pilot / f"{task}.summary.json"
        if summ.exists():
            s = json.loads(summ.read_text())
            lines += [f"Chosen budget: {json.dumps(s['chosen']['entry'])} with pilot rate {s['chosen']['rate']:.2f}"
                      + (f" (flag: {s['chosen']['flag']})" if s["chosen"]["flag"] else "") + ".", ""]
        else:
            lines += ["No budget chosen yet.", ""]
        cur = sorted(pilot.glob(f"{task}.*.jsonl"))
        if cur:
            lines += ["Grid entries with the final settings:", "", table(cur, fmt[task]), ""]
        old = sorted((pilot / "original_grid").glob(f"{task}.*.jsonl"))
        if old:
            lines += ["Entries of the original grid, superseded by an amendment (earlier settings, partial entries included):", "",
                      table(old, fmt[task]), ""]
        exp = sorted((pilot / "explore").glob(f"{task}.*.jsonl"))
        if exp:
            lines += ["Exploration runs (a few pilot seeds each):", "", table(exp, fmt[task]), ""]
    (ROOT / "docs" / "PILOT.md").write_text("\n".join(lines).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
