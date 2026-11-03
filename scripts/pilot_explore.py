"""Pilot exploration on pilot seeds only (reference environment). Used when no entry of the
pre registered grid reaches the target band. Every call is kept under results/pilot/explore/.

Example: python scripts/pilot_explore.py --task grok --set p=13 lr=0.003 steps=4000 train_frac=0.5 --seeds 1000:1004
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
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seeds", default="1000:1004", help="start:stop, stop excluded; pilot seeds are 1000 to 1009")
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split(":"))
    assert lo >= 1000 and hi <= 1010, "exploration must use pilot seeds only"
    cfg = load_cfg(args.task, need_budget=False)
    over = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    cfg.update(over)
    key = args.task + "." + "_".join(f"{k}{v}" for k, v in sorted(over.items()))
    out = ROOT / "results" / "pilot" / "explore" / f"{key}.jsonl"
    fn = runner_for(args.task)
    table = PairedRunner(lambda s, e: fn(s, e, cfg), [ARMS["ref"]], out).run(range(lo, hi))
    rows = table.rows("ref")
    print(key, "rate", sum(r["outcome"] for r in rows.values()) / len(rows), "scores", [round(r["score"], 3) for r in rows.values()])


if __name__ == "__main__":
    main()
