"""Analysis checks on hand built rows. The data below is SYNTHETIC and exists only to test the code."""
import json
import math

from thabat_study import analysis


def write(tmp_path, task, rows):
    (tmp_path / f"{task}.a.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def row(env, seed, score, thr, h):
    return {"env": env, "seed": seed, "score": score, "outcome": int(score >= thr), "param_hash": h}


def test_metrics_flip_perturbation_and_h3_on_synthetic_rows(tmp_path):
    thr = -250.0
    rows = []
    # SYNTHETIC: 10 seeds; ref scores spread around the threshold; fp64 flips the two seeds nearest to it
    ref_scores = [-600, -500, -400, -300, -260, -240, -200, -150, -100, -50]
    for s, sc in enumerate(ref_scores):
        arm = sc + (30 if s == 4 else -30 if s == 5 else 1.0)
        rows += [row("ref", s, sc, thr, f"r{s}"), row("ref_repeat", s, sc, thr, f"r{s}"),
                 row("fp64", s, arm, thr, f"f{s}"), row("thr", s, sc, thr, f"r{s}")]
    write(tmp_path, "pendulum", rows)
    res = analysis.analyse(tmp_path, {"pendulum": thr, "cartpole": 0, "invpend": 0}, n_boot=200)
    m = res["metrics"]
    assert m["n_seeds.pendulum"] == 10 and m["determinism_rate.pendulum"] == 1.0
    assert m["flip_rate.pendulum.fp64"] == 0.2 and m["perturbation_rate.pendulum.fp64"] == 1.0
    assert m["flip_rate.pendulum.thr"] == 0.0 and m["perturbation_rate.pendulum.thr"] == 0.0
    assert m["abs_delta.pendulum.fp64"] == 0.0 and m["mcnemar_p.pendulum.fp64"] == 1.0
    assert m["mcnemar_holm_p.pendulum.fp64"] == 1.0
    # H3: only the arm with a valid perturbation check is pooled, and the flipped seeds are the closest ones
    assert m["n_flips.pendulum"] == 2 and m["auc_dist.pendulum"] == 1.0
    assert "flip_rate.pendulum.gpu" not in m  # arm not run, so the scorer will say NOT RUN


def test_no_flips_gives_nan_auc_and_zero_count(tmp_path):
    thr = 0.0
    rows = []
    for s in range(4):  # SYNTHETIC
        rows += [row("ref", s, 1.0 + s, thr, f"r{s}"), row("fp64", s, 1.0 + s, thr, f"f{s}")]
    write(tmp_path, "cartpole", rows)
    m = analysis.analyse(tmp_path, {"pendulum": 0, "cartpole": thr, "invpend": 0}, n_boot=100)["metrics"]
    assert m["n_flips.cartpole"] == 0 and math.isnan(m["auc_dist.cartpole"])


def test_gpu_uses_same_machine_reference_and_jsonable(tmp_path):
    thr = 0.0
    rows = []
    for s in range(6):  # SYNTHETIC
        rows += [row("ref", s, 1.0, thr, "a"), row("ref_k", s, 1.0 if s else -1.0, thr, "b"),
                 row("gpu", s, 1.0 if s else -1.0, thr, "c")]
    write(tmp_path, "invpend", rows)
    res = analysis.analyse(tmp_path, {"pendulum": 0, "cartpole": 0, "invpend": thr}, n_boot=100)
    assert res["metrics"]["flip_rate.invpend.gpu"] == 0.0  # compared with ref_k, not ref
    assert res["metrics"]["flip_rate.invpend.machine"] == 1 / 6
    out = tmp_path / "s.json"
    analysis.write_summary(res, out)
    assert "invpend.gpu" in json.loads(out.read_text())["comps"]
