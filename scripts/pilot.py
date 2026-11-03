"""Choose training budgets on pilot seeds only (reference environment).

Rule (docs/PREDICTIONS.md): walk the grid in the listed order, stop at the first
entry whose pilot success rate is in [0.3, 0.7]; otherwise take the entry closest
to 0.5 and flag it. Writes configs/budget.json and results/pilot/<task>.summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thabat_kit.runner import PairedRunner  # noqa: E402
from thabat_study.tasks import ARMS, load_cfg, runner_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    grid = json.loads((ROOT / "configs" / "pilot_grid.json").read_text())
    lo, hi = grid["pilot_seeds"]
    base = load_cfg(args.task, need_budget=False)
    fn = runner_for(args.task)
    out_dir = ROOT / "results" / "pilot"
    tried = []
    chosen = None
    for i, entry in enumerate(grid[args.task]):
        cfg = {**base, **entry}
        table = PairedRunner(lambda s, e: fn(s, e, cfg), [ARMS["ref"]], out_dir / f"{args.task}.{i}.jsonl").run(
            range(lo, hi + 1), log=print
        )
        rows = table.rows("ref")
        rate = sum(r["outcome"] for r in rows.values()) / len(rows)
        tried.append({"entry": entry, "rate": rate, "n": len(rows)})
        print(args.task, entry, "pilot rate", rate, flush=True)
        if 0.3 <= rate <= 0.7:
            chosen = {"entry": entry, "rate": rate, "flag": None}
            break
    if chosen is None:
        best = min(tried, key=lambda t: abs(t["rate"] - 0.5))
        chosen = {"entry": best["entry"], "rate": best["rate"], "flag": "no grid entry reached the target band; closest to 0.5 used"}
    (out_dir / f"{args.task}.summary.json").write_text(json.dumps({"tried": tried, "chosen": chosen}, indent=2))
    budget_path = ROOT / "configs" / "budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    budget[args.task] = chosen["entry"]
    budget_path.write_text(json.dumps(budget, indent=2, sort_keys=True))
    print("chosen", args.task, chosen)


if __name__ == "__main__":
    main()
