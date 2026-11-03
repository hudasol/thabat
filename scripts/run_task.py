"""Run one task under chosen arms and seeds. Appends to results/raw/<task>.<shard>.jsonl.

Example: python scripts/run_task.py --task pendulum --arms ref,thr,fp64 --seeds 0:15 --shard a
"""
import argparse
import os
import sys
from pathlib import Path

# MKL reads this when the library loads, so it must be set before torch is imported.
if "--mkl-cbwr" in sys.argv:
    os.environ["MKL_CBWR"] = sys.argv[sys.argv.index("--mkl-cbwr") + 1]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thabat_kit.runner import PairedRunner  # noqa: E402
from thabat_study.tasks import ARMS, load_cfg, runner_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arms", default="ref,thr,fp64")
    ap.add_argument("--seeds", default="0:30", help="start:stop, stop excluded")
    ap.add_argument("--shard", default="a")
    ap.add_argument("--out", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--mkl-cbwr", default=None, help="set MKL_CBWR (needed for the isa arm)")
    args = ap.parse_args()
    if "isa" in args.arms.split(",") and os.environ.get("MKL_CBWR") != "COMPATIBLE":
        sys.exit("the isa arm needs --mkl-cbwr COMPATIBLE")
    if os.environ.get("MKL_CBWR") and "isa" not in args.arms.split(","):
        sys.exit("MKL_CBWR is set, so only the isa arm may run in this process")
    lo, hi = map(int, args.seeds.split(":"))
    cfg = load_cfg(args.task)
    fn = runner_for(args.task)
    envs = [ARMS[a] for a in args.arms.split(",")]
    path = Path(args.out) / f"{args.task}.{args.shard}.jsonl"
    PairedRunner(lambda seed, env: fn(seed, env, cfg), envs, path).run(range(lo, hi), log=print)


if __name__ == "__main__":
    main()
