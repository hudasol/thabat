"""Render docs/RESULTS.md and docs/PAPER.md from their templates and results/summary.json.

Every number in those documents comes from the summary file or from tables written by
scripts/analyze.py. tests/test_docs_current.py fails if a committed document differs
from a fresh rendering.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thabat_kit import report, scorer  # noqa: E402

ARM_NAME = {"thr": "4 threads", "fp64": "float64", "isa": "kernel path", "gpu": "GPU", "machine": "other machine"}


def f(x, spec=".3f"):
    return "n/a" if x in ("", None) or x != x else format(x, spec)


def blocks(root: Path) -> dict:
    summary = json.loads((root / "results" / "summary.json").read_text())
    comps, metrics = summary["comps"], summary["metrics"]
    rows = ["| Task | Arm | n | Reference rate | Arm rate | Lost | Gained | Flip rate (95% CI) | McNemar p | Holm p | Bits differ |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(comps):
        c = comps[key]
        task, arm = key.split(".")
        holm = metrics.get(f"mcnemar_holm_p.{key}")
        rows.append(f"| {task} | {ARM_NAME.get(arm, arm)} | {c['n']} | {f(c['ref_rate'], '.3f')} | {f(c['arm_rate'], '.3f')} | {c['lost']} | {c['gained']} | "
                    f"{f(c['flip_rate'], '.3f')} [{f(c['flip_ci'][0])}, {f(c['flip_ci'][1])}] | {f(c['mcnemar_p'])} | {f(holm)} | {f(c.get('perturbation_rate'))} |")
    h3 = ["| Task | Pooled pairs | Flipped | AUC of distance (95% CI) | One sided p |", "|---|---|---|---|---|"]
    for t in ("pendulum", "cartpole", "invpend", "all"):
        if f"n_flips.{t}" in metrics:
            h3.append(f"| {t} | {metrics[f'n_pairs.{t}']} | {metrics[f'n_flips.{t}']} | {f(metrics.get(f'auc_dist.{t}'))} "
                      f"[{f(metrics.get(f'auc_lo.{t}'))}, {f(metrics.get(f'auc_hi.{t}'))}] | {f(metrics.get(f'auc_p.{t}'))} |")
    preds = scorer.parse_predictions((root / "docs" / "PREDICTIONS.md").read_text())
    scores = scorer.score(preds, metrics)
    counts = scorer.summarise(scores)
    envs = {}
    for p in sorted((root / "results" / "raw").glob("*.envs.json")):
        for name, cap in json.loads(p.read_text()).items():
            envs.setdefault(name, cap)
    erows = ["| Arm | CPU | Torch | Threads | BLAS | MKL_CBWR | Device | Dtype |", "|---|---|---|---|---|---|---|---|"]
    for name, cap in envs.items():
        r = cap["requested"]
        t = cap["torch"] or {}
        erows.append(f"| {name} | {cap.get('cpu_model')} | {t.get('version')} | {t.get('threads')} | {cap.get('blas')} | "
                     f"{cap['env_vars'].get('MKL_CBWR')} | {t.get('gpu_name') or r['device']} | {r['dtype']} |")
    out = {"block.summary": "\n".join(rows), "block.h3": "\n".join(h3), "block.predictions": scorer.markdown_table(scores),
           "block.envs": "\n".join(erows),
           "count.hit": counts.get("HIT", 0), "count.miss": counts.get("MISS", 0),
           "count.not_run": counts.get("NOT RUN", 0), "count.not_testable": counts.get("NOT TESTABLE", 0),
           "count.total": len(scores)}
    libs = next(iter(envs.values()))["libraries"] if envs else {}
    for k, v in libs.items():
        out[f"lib.{k}"] = v
    return {**metrics, **out}


def main(root: Path = ROOT) -> None:
    ctx = blocks(root)
    for name in ("RESULTS", "PAPER"):
        tpl = root / "docs" / f"{name}.template.md"
        if tpl.exists():
            report.render_file(tpl, ctx, root / "docs" / f"{name}.md")
            print("rendered", name)


if __name__ == "__main__":
    main()
