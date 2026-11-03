"""Raw rows -> results/summary.json, tables/*.csv, figures/*.png, tables/predictions.csv."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thabat_kit import scorer  # noqa: E402
from thabat_study import analysis, figures  # noqa: E402
from thabat_study.tasks import TASKS, load_cfg  # noqa: E402


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(root: Path = ROOT) -> dict:
    raw, tables, figs = root / "results" / "raw", root / "tables", root / "figures"
    tables.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    thresholds = {t: load_cfg(t, need_budget=False)["threshold"] for t in TASKS}
    res = analysis.analyse(raw, thresholds)
    build = root / "results" / "build.json"
    if build.exists():
        res["metrics"].update(json.loads(build.read_text()))
    analysis.write_summary(res, root / "results" / "summary.json")

    rows = []
    for (task, arm), c in sorted(res["comps"].items()):
        holm = res["metrics"].get(f"mcnemar_holm_p.{task}.{arm}")
        rows.append({"task": task, "arm": arm, "n": c["n"], "ref_rate": round(c["ref_rate"], 4),
                     "arm_rate": round(c["arm_rate"], 4), "lost": c["lost"], "gained": c["gained"], "flips": c["flips"],
                     "flip_rate": round(c["flip_rate"], 4), "flip_lo": round(c["flip_ci"][0], 4), "flip_hi": round(c["flip_ci"][1], 4),
                     "delta": round(c["delta"], 4), "delta_lo": round(c["delta_ci"][0], 4), "delta_hi": round(c["delta_ci"][1], 4),
                     "mcnemar_p": round(c["mcnemar_p"], 4), "mcnemar_holm_p": "" if holm is None else round(holm, 4),
                     "perturbation_rate": round(c.get("perturbation_rate", float("nan")), 4)})
    write_csv(tables / "summary.csv", rows, list(rows[0])) if rows else None
    if res["pairs"]:
        write_csv(tables / "paired_scores.csv", res["pairs"], list(res["pairs"][0]))

    comps = {k: v for k, v in res["comps"].items()}
    if comps:
        figures.flip_rates(comps, figs / "flip_rates.png")
        figures.success_rates(comps, figs / "success_rates.png")
    if res["pairs"]:
        figures.paired_scatter(res["pairs"], thresholds, figs / "paired_scatter.png")
    from thabat_kit.runner import PairedTable
    g = analysis.load_task(raw, "grok")
    gc = res["comps"].get(("grok", "thr"))
    if gc and "curve" in next(iter(g.rows("ref").values()), {}):
        flipped = [s for s in gc["seeds"] if g.rows("ref")[s]["outcome"] != g.rows("thr")[s]["outcome"]]
        figures.grok_curves(g, flipped, thresholds["grok"], figs / "grok_curves.png")

    preds = scorer.parse_predictions((root / "docs" / "PREDICTIONS.md").read_text())
    scores = scorer.score(preds, res["metrics"])
    write_csv(tables / "predictions.csv",
              [{"id": s.id, "status": s.status, "checks": "; ".join(map(str, s.prediction.checks)),
                "observed": "; ".join(f"{k}={v}" for k, v in s.observed.items())} for s in scores],
              ["id", "status", "checks", "observed"])
    print(scorer.summarise(scores))
    return res


if __name__ == "__main__":
    main()
